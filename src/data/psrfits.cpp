#include "psrfits.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fitsio.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>

using namespace std;

namespace {

void check_fits_status(int status, const string &context) {
  if (status == 0) {
    return;
  }
  char message[FLEN_STATUS];
  fits_get_errstatus(status, message);
  throw runtime_error(context + ": " + message);
}

template <typename T>
void read_key_or_default(fitsfile *fptr, int datatype, const char *key, T *value,
                         T fallback) {
  int status = 0;
  fits_read_key(fptr, datatype, key, value, nullptr, &status);
  if (status != 0) {
    *value = fallback;
  }
}

string read_string_key_or_default(fitsfile *fptr, const char *key,
                                  const string &fallback = "") {
  char value[FLEN_VALUE] = {0};
  int status = 0;
  fits_read_key(fptr, TSTRING, key, value, nullptr, &status);
  if (status != 0) {
    return fallback;
  }
  return string(value);
}

template <typename T>
void combine_polarizations(const vector<T> &raw, T *dst, long row_offset,
                           long nsblk, int npol, int nchans,
                           bool flip_freq) {
  for (long sample = 0; sample < nsblk; ++sample) {
    const long sample_base = sample * npol * nchans;
    T *out = dst + (row_offset + sample) * nchans;
    for (int ch = 0; ch < nchans; ++ch) {
      const int src_ch = flip_freq ? (nchans - 1 - ch) : ch;
      const long idx0 = sample_base + src_ch;
      if (npol == 1) {
        out[ch] = raw[idx0];
      } else {
        const long idx1 = sample_base + nchans + src_ch;
        out[ch] = static_cast<T>((static_cast<uint64_t>(raw[idx0]) +
                                  static_cast<uint64_t>(raw[idx1])) /
                                 2);
      }
    }
  }
}

void combine_float32_polarizations(const vector<float> &raw, uint32_t *dst,
                                   long row_offset, long nsblk, int npol,
                                   int nchans, bool flip_freq) {
  for (long sample = 0; sample < nsblk; ++sample) {
    const long sample_base = sample * npol * nchans;
    uint32_t *out = dst + (row_offset + sample) * nchans;
    for (int ch = 0; ch < nchans; ++ch) {
      const int src_ch = flip_freq ? (nchans - 1 - ch) : ch;
      const long idx0 = sample_base + src_ch;
      double value = raw[idx0];
      if (npol > 1) {
        const long idx1 = sample_base + nchans + src_ch;
        value = 0.5 * (value + raw[idx1]);
      }
      value = std::clamp(value, 0.0,
                         static_cast<double>(numeric_limits<uint32_t>::max()));
      out[ch] = static_cast<uint32_t>(llround(value));
    }
  }
}

} // namespace

PsrFits::PsrFits()
    : filename(""), nifs(1), npol(0), nbits(0), nchans(0), nsubint(0),
      nsblk(0), ndata(0), mjd(0.0), tsamp(0.0), fch1(0.0), foff(0.0),
      raj(""), decj(""), data(nullptr), data_owner(nullptr), flip_freq_(false) {}

PsrFits::PsrFits(const string &fname) : PsrFits() {
  filename = fname;
  read_header();
  read_data();
}

PsrFits::~PsrFits() { close(); }

void PsrFits::close() {
  data_owner.reset();
  data = nullptr;
}

bool PsrFits::read_header() {
  if (filename.empty()) {
    throw runtime_error("PSRFITS filename is empty");
  }

  fitsfile *fptr = nullptr;
  int status = 0;
  fits_open_file(&fptr, filename.c_str(), READONLY, &status);
  check_fits_status(status, "failed to open PSRFITS file");

  double obsfreq = 0.0;
  double obsbw = 0.0;
  long stt_imjd = 0;
  long stt_smjd = 0;
  double stt_offs = 0.0;

  fits_read_key(fptr, TDOUBLE, "OBSFREQ", &obsfreq, nullptr, &status);
  check_fits_status(status, "failed to read OBSFREQ");
  fits_read_key(fptr, TDOUBLE, "OBSBW", &obsbw, nullptr, &status);
  check_fits_status(status, "failed to read OBSBW");
  read_key_or_default(fptr, TLONG, "STT_IMJD", &stt_imjd, 0L);
  read_key_or_default(fptr, TLONG, "STT_SMJD", &stt_smjd, 0L);
  read_key_or_default(fptr, TDOUBLE, "STT_OFFS", &stt_offs, 0.0);
  raj = read_string_key_or_default(fptr, "RA");
  decj = read_string_key_or_default(fptr, "DEC");
  mjd = static_cast<double>(stt_imjd) +
        (static_cast<double>(stt_smjd) + stt_offs) / 86400.0;

  fits_movnam_hdu(fptr, BINARY_TBL, const_cast<char *>("SUBINT"), 0, &status);
  check_fits_status(status, "failed to move to SUBINT HDU");

  long rows = 0;
  fits_get_num_rows(fptr, &rows, &status);
  check_fits_status(status, "failed to read SUBINT row count");
  nsubint = rows;

  fits_read_key(fptr, TINT, "NCHAN", &nchans, nullptr, &status);
  check_fits_status(status, "failed to read NCHAN");
  fits_read_key(fptr, TINT, "NPOL", &npol, nullptr, &status);
  check_fits_status(status, "failed to read NPOL");
  fits_read_key(fptr, TINT, "NBITS", &nbits, nullptr, &status);
  check_fits_status(status, "failed to read NBITS");
  fits_read_key(fptr, TLONG, "NSBLK", &nsblk, nullptr, &status);
  check_fits_status(status, "failed to read NSBLK");
  fits_read_key(fptr, TDOUBLE, "TBIN", &tsamp, nullptr, &status);
  check_fits_status(status, "failed to read TBIN");
  fits_read_key(fptr, TDOUBLE, "CHAN_BW", &foff, nullptr, &status);
  check_fits_status(status, "failed to read CHAN_BW");

  if (npol < 1) {
    throw runtime_error("Unsupported NPOL value: " + to_string(npol));
  }
  if (nbits != 8 && nbits != 16 && nbits != 32) {
    throw runtime_error("Unsupported PSRFITS NBITS value: " +
                        to_string(nbits) +
                        ". Only 8/16/32 bit data is supported.");
  }

  nifs = 1;
  ndata = nsubint * nsblk;
  fch1 = obsfreq - obsbw / 2.0;
  flip_freq_ = foff < 0;
  if (flip_freq_) {
    foff = -foff;
    fch1 = fch1 - (nchans - 1) * foff;
  }

  fits_close_file(fptr, &status);
  check_fits_status(status, "failed to close PSRFITS file");
  return true;
}

bool PsrFits::read_data() {
  if (nsubint <= 0 || nsblk <= 0 || nchans <= 0 || npol <= 0) {
    read_header();
  }

  fitsfile *fptr = nullptr;
  int status = 0;
  fits_open_file(&fptr, filename.c_str(), READONLY, &status);
  check_fits_status(status, "failed to open PSRFITS file");
  fits_movnam_hdu(fptr, BINARY_TBL, const_cast<char *>("SUBINT"), 0, &status);
  check_fits_status(status, "failed to move to SUBINT HDU");

  int data_col = 0;
  fits_get_colnum(fptr, CASEINSEN, const_cast<char *>("DATA"), &data_col,
                  &status);
  check_fits_status(status, "failed to find DATA column");

  int typecode = 0;
  long repeat = 0;
  long width = 0;
  fits_get_coltype(fptr, data_col, &typecode, &repeat, &width, &status);
  check_fits_status(status, "failed to inspect DATA column");
  const long expected_repeat = nsblk * npol * nchans;
  if (repeat != expected_repeat) {
    throw runtime_error("Unsupported PSRFITS DATA repeat: " +
                        to_string(repeat) + ", expected " +
                        to_string(expected_repeat));
  }

  const long output_elements = ndata * nchans;
  switch (nbits) {
  case 8: {
    auto owner = shared_ptr<uint8_t[]>(new uint8_t[output_elements]);
    data_owner = owner;
    data = owner.get();
    vector<uint8_t> raw(expected_repeat);
    int anynul = 0;
    for (long row = 1; row <= nsubint; ++row) {
      fits_read_col(fptr, TBYTE, data_col, row, 1, expected_repeat, nullptr,
                    raw.data(), &anynul, &status);
      check_fits_status(status, "failed to read 8-bit DATA column");
      combine_polarizations(raw, owner.get(), (row - 1) * nsblk, nsblk, npol,
                            nchans, flip_freq_);
    }
    break;
  }
  case 16: {
    auto owner = shared_ptr<uint16_t[]>(new uint16_t[output_elements]);
    data_owner = owner;
    data = owner.get();
    vector<uint16_t> raw(expected_repeat);
    int anynul = 0;
    for (long row = 1; row <= nsubint; ++row) {
      fits_read_col(fptr, TUSHORT, data_col, row, 1, expected_repeat, nullptr,
                    raw.data(), &anynul, &status);
      check_fits_status(status, "failed to read 16-bit DATA column");
      combine_polarizations(raw, owner.get(), (row - 1) * nsblk, nsblk, npol,
                            nchans, flip_freq_);
    }
    break;
  }
  case 32: {
    auto owner = shared_ptr<uint32_t[]>(new uint32_t[output_elements]);
    data_owner = owner;
    data = owner.get();
    int anynul = 0;
    if (typecode == TFLOAT) {
      vector<float> raw(expected_repeat);
      for (long row = 1; row <= nsubint; ++row) {
        fits_read_col(fptr, TFLOAT, data_col, row, 1, expected_repeat, nullptr,
                      raw.data(), &anynul, &status);
        check_fits_status(status, "failed to read float32 DATA column");
        combine_float32_polarizations(raw, owner.get(), (row - 1) * nsblk,
                                      nsblk, npol, nchans, flip_freq_);
      }
    } else {
      vector<uint32_t> raw(expected_repeat);
      for (long row = 1; row <= nsubint; ++row) {
        fits_read_col(fptr, TUINT, data_col, row, 1, expected_repeat, nullptr,
                      raw.data(), &anynul, &status);
        check_fits_status(status, "failed to read 32-bit DATA column");
        combine_polarizations(raw, owner.get(), (row - 1) * nsblk, nsblk, npol,
                              nchans, flip_freq_);
      }
    }
    break;
  }
  default:
    throw runtime_error("Unsupported nbits value in read_data");
  }

  fits_close_file(fptr, &status);
  check_fits_status(status, "failed to close PSRFITS file");
  return true;
}

variant<uint8_t *, uint16_t *, uint32_t *> PsrFits::get_data(int idx) {
  if (idx >= ndata) {
    throw runtime_error("index out of range in PsrFits::get_data");
  }
  if (data == nullptr) {
    throw runtime_error("data is null in PsrFits::get_data");
  }

  const long offset = idx * nchans;
  switch (nbits) {
  case 8:
    return static_cast<uint8_t *>(data) + offset;
  case 16:
    return static_cast<uint16_t *>(data) + offset;
  case 32:
    return static_cast<uint32_t *>(data) + offset;
  default:
    throw runtime_error("Unsupported nbits value in get_data");
  }
}

void PsrFits::info() const {
  cout << "PSRFITS Information" << endl;
  cout << "-------------------" << endl;
  cout << left << setw(20) << "Filename:" << filename << endl;
  cout << left << setw(20) << "MJD:" << mjd << endl;
  cout << left << setw(20) << "RAJ:" << raj << endl;
  cout << left << setw(20) << "DECJ:" << decj << endl;
  cout << left << setw(20) << "Sample Time:" << tsamp << " s" << endl;
  cout << left << setw(20) << "Number of Bits:" << nbits << endl;
  cout << left << setw(20) << "Number of Channels:" << nchans << endl;
  cout << left << setw(20) << "Input Polarizations:" << npol << endl;
  cout << left << setw(20) << "Output IFs:" << nifs << endl;
  cout << left << setw(20) << "SUBINT Rows:" << nsubint << endl;
  cout << left << setw(20) << "Samples per Row:" << nsblk << endl;
  cout << left << setw(20) << "Total Samples:" << ndata << endl;
  cout << left << setw(20) << "First Channel Freq:" << fch1 << " MHz" << endl;
  cout << left << setw(20) << "Channel Bandwidth:" << foff << " MHz" << endl;
  cout << "-------------------" << endl;
}

template <typename T> shared_ptr<T[]> PsrFits::get_shared_ptr_data() {
  if ((typeid(T) == typeid(uint8_t) && nbits != 8) ||
      (typeid(T) == typeid(uint16_t) && nbits != 16) ||
      (typeid(T) == typeid(uint32_t) && nbits != 32)) {
    throw runtime_error("Template type does not match nbits value");
  }
  return shared_ptr<T[]>(data_owner, static_cast<T *>(data));
}

template shared_ptr<uint8_t[]> PsrFits::get_shared_ptr_data<uint8_t>();
template shared_ptr<uint16_t[]> PsrFits::get_shared_ptr_data<uint16_t>();
template shared_ptr<uint32_t[]> PsrFits::get_shared_ptr_data<uint32_t>();
