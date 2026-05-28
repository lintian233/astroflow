#pragma once

#ifndef PSRFITS_READER_H_
#define PSRFITS_READER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <variant>

class PsrFits {
public:
  PsrFits();
  explicit PsrFits(const std::string &fname);
  ~PsrFits();

  bool read_header();
  bool read_data();
  void close();
  void info() const;

  std::variant<uint8_t *, uint16_t *, uint32_t *> get_data(int idx);
  template <typename T> std::shared_ptr<T[]> get_shared_ptr_data();

public:
  std::string filename;
  int nifs;
  int npol;
  int nbits;
  int nchans;
  long int nsubint;
  long int nsblk;
  long int ndata;
  double mjd;
  double tsamp;
  double fch1;
  double foff;
  void *data;
  std::shared_ptr<void> data_owner;

private:
  bool flip_freq_;
};

#endif // PSRFITS_READER_H_
