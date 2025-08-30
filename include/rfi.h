#pragma once 

#ifndef _RFI_H
#define _RFI_H

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

struct iqrmconfig
{
    iqrmconfig()
        : mode(0), radius_frac(0.10f), nsigma(3.0f), geofactor(1.5),
          win_sec(5.0), hop_sec(2.5), include_tail(true) {}
    iqrmconfig(
        int mode, float radius_frac, float nsigma, double geofactor,
        double win_sec, double hop_sec, bool include_tail)
        : mode(mode), radius_frac(radius_frac), nsigma(nsigma), geofactor(geofactor),
          win_sec(win_sec), hop_sec(hop_sec), include_tail(include_tail) {}
    iqrmconfig(py::object cfg)
    {
        mode = cfg.attr("mode").cast<int>();
        radius_frac = cfg.attr("radius_frac").cast<float>();
        nsigma = cfg.attr("nsigma").cast<float>();
        geofactor = cfg.attr("geofactor").cast<double>();
        win_sec = cfg.attr("win_sec").cast<double>();
        hop_sec = cfg.attr("hop_sec").cast<double>();
        include_tail = cfg.attr("include_tail").cast<bool>();
    }
    int mode;
    float radius_frac;
    float nsigma;
    double geofactor;
    double win_sec;
    double hop_sec;
    bool include_tail;
};

struct rficonfig
{   
    rficonfig() : use_mask(false), use_zero_dm(false), use_iqrm(false), iqrm_cfg() {}
    rficonfig(bool use_mask, bool use_zero_dm, bool use_iqrm, iqrmconfig iqrm_cfg)
        : use_mask(use_mask), use_zero_dm(use_zero_dm), use_iqrm(use_iqrm), iqrm_cfg(iqrm_cfg) {}
    rficonfig(py::object cfg)
    {
        use_zero_dm = cfg.attr("use_zero_dm").cast<bool>();
        use_mask = cfg.attr("use_mask").cast<bool>();
        use_iqrm = cfg.attr("use_iqrm").cast<bool>();
        iqrm_cfg = iqrmconfig(cfg.attr("iqrm_cfg"));
    }
    bool use_zero_dm;
    bool use_mask;
    bool use_iqrm;
    iqrmconfig iqrm_cfg;
};




#endif // _RFI_H