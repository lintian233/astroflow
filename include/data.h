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

#endif //_DATA_H
