/**
 * Project astroflow
 */

#ifndef _DATA_H
#define _DATA_H

#include <memory>
#include <string>
#include <vector>

class data {};

struct dedisperseddata {
  std::vector<std::shared_ptr<uint32_t[]>> dm_times;

  //[DM,downtsample]
  std::vector<size_t> shape;
  int dm_ndata;
  int downtsample_ndata;

  // dm_low, dm_high, dm_step, dm_size
  float dm_low;
  float dm_high;
  float dm_step;
  float tsample;
  // filname
  std::string filname;
};

template <typename T> struct Spectrum {
  std::shared_ptr<T[]> data;
  size_t nchans;
  size_t ntimes;
  float tstart;
  float tend;
  float dm;
  int nbits;
  float freq_start;
  float freq_end;
};

#endif //_DATA_H
