/*
 * filterbank.h
 *
 *  Created on: Feb 19, 2020
 *      Author: ypmen
 *
 *  Fixed on: Jan 26, 2025
 *      Author: (xd)[https://github.com/lintian233]
 *
 *  Memory-Mapped I/O Optimization: Apr 29, 2026
 */

#ifndef FILTERBANK_H_
#define FILTERBANK_H_

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdio.h>
#include <string>
#include <variant>
#include <vector>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <iostream>

using namespace std;

/**
 * @class MMapBuffer
 * @brief Zero-copy memory-mapped file buffer for efficient large file access
 * 
 * Automatically handles zero-copy access to disk data, minimal CPU usage,
 * supports random access at O(1) time, and leverages system page cache.
 */
class MMapBuffer {
private:
    int fd_ = -1;
    void* mmap_ptr_ = nullptr;
    size_t mmap_size_ = 0;
    size_t file_offset_ = 0;
    
public:
    MMapBuffer() = default;
    ~MMapBuffer() { cleanup(); }
    
    MMapBuffer(const MMapBuffer&) = delete;
    MMapBuffer& operator=(const MMapBuffer&) = delete;
    
    MMapBuffer(MMapBuffer&& other) noexcept 
        : fd_(other.fd_), mmap_ptr_(other.mmap_ptr_), 
          mmap_size_(other.mmap_size_), file_offset_(other.file_offset_) {
        other.fd_ = -1;
        other.mmap_ptr_ = nullptr;
    }
    
    MMapBuffer& operator=(MMapBuffer&& other) noexcept {
        cleanup();
        fd_ = other.fd_;
        mmap_ptr_ = other.mmap_ptr_;
        mmap_size_ = other.mmap_size_;
        file_offset_ = other.file_offset_;
        other.fd_ = -1;
        other.mmap_ptr_ = nullptr;
        return *this;
    }
    
    /**
     * @brief Map file starting from given offset
     * @param filename Path to file
     * @param offset File offset to start mapping (typically header_size)
     * @param size Size to map (0 = rest of file)
     * @return true on success, false otherwise
     */
    bool map(const std::string& filename, size_t offset, size_t size = 0) {
        cleanup();
        
        fd_ = open(filename.c_str(), O_RDONLY | O_NOATIME);
        if (fd_ < 0) {
            std::cerr << "Failed to open " << filename << std::endl;
            return false;
        }
        
        struct stat sb;
        if (fstat(fd_, &sb) < 0) {
            std::cerr << "Failed to stat " << filename << std::endl;
            close(fd_);
            fd_ = -1;
            return false;
        }
        
        if (offset >= (size_t)sb.st_size) {
            std::cerr << "Offset exceeds file size" << std::endl;
            close(fd_);
            fd_ = -1;
            return false;
        }
        
        mmap_size_ = (size == 0) ? (sb.st_size - offset) : size;
        file_offset_ = offset;
        
        // Try to use huge pages + populate for maximum speed
        // Fallback to regular mapping if huge pages not available
        int mmap_flags = MAP_SHARED | MAP_POPULATE;
#ifdef MAP_HUGETLB
        mmap_flags |= MAP_HUGETLB;
#endif
        
        mmap_ptr_ = mmap(nullptr, mmap_size_, PROT_READ, mmap_flags, fd_, offset);
        
        // Fallback if huge pages fail
        if (mmap_ptr_ == MAP_FAILED && (mmap_flags & MAP_HUGETLB)) {
            mmap_flags &= ~MAP_HUGETLB;
            mmap_ptr_ = mmap(nullptr, mmap_size_, PROT_READ, mmap_flags, fd_, offset);
        }
        
        if (mmap_ptr_ == MAP_FAILED) {
            std::cerr << "mmap failed" << std::endl;
            close(fd_);
            fd_ = -1;
            mmap_ptr_ = nullptr;
            return false;
        }
        
        // Optimized memory advise for fast sequential access + minimal CPU
        // MADV_SEQUENTIAL: optimize for sequential access
        // MADV_WILLNEED: async read ahead (doesn't block)
        // MADV_HUGEPAGE: encourage huge page usage
        madvise(mmap_ptr_, mmap_size_, MADV_SEQUENTIAL | MADV_WILLNEED);
#ifdef MADV_HUGEPAGE
        madvise(mmap_ptr_, mmap_size_, MADV_HUGEPAGE);
#endif
        
        return true;
    }
    
    /**
     * @brief Get pointer to data at given offset within mapped region
     */
    template <typename T>
    T* get(size_t offset = 0) const {
        if (!mmap_ptr_) return nullptr;
        if (offset >= mmap_size_) return nullptr;
        // Carefully handle pointer arithmetic on void*
        return reinterpret_cast<T*>(static_cast<char*>(mmap_ptr_) + offset * sizeof(T));
    }
    
    /**
     * @brief Check if buffer is valid
     */
    bool valid() const { return mmap_ptr_ != nullptr; }
    
    /**
     * @brief Get total mapped size in bytes
     */
    size_t size() const { return mmap_size_; }
    
    /**
     * @brief Pre-fault memory for maximum performance (optional)
     */
    void prefault() {
        if (!mmap_ptr_) return;
        volatile uint8_t* ptr = static_cast<uint8_t*>(mmap_ptr_);
        for (size_t i = 0; i < mmap_size_; i += 4096) {
            (void)*ptr;
            ptr += 4096;
        }
    }
    
private:
    void cleanup() {
        if (mmap_ptr_ && mmap_ptr_ != MAP_FAILED) {
            munmap(mmap_ptr_, mmap_size_);
            mmap_ptr_ = nullptr;
        }
        if (fd_ >= 0) {
            close(fd_);
            fd_ = -1;
        }
    }
};

class Filterbank {
public:
  Filterbank();
  Filterbank(const Filterbank &fil);
  Filterbank &operator=(const Filterbank &fil);
  Filterbank(const string fname);
  ~Filterbank();
  void free();
  void close();
  bool read_header();
  bool read_data();
  bool read_data_mmap();  // Memory-mapped I/O version
  bool set_data(unsigned char *dat, long int ns, int nif, int nchan);
  bool write_header();
  bool write_data();

  variant<uint8_t *, uint16_t *, uint32_t *> get_data(int idx);
  template <typename T> bool read_data_impl();
  template <typename T> bool read_data_mmap_impl();  // Memory-mapped template
  void info() const;
  template <typename T> std::shared_ptr<T[]> get_shared_ptr_data();
  bool is_mmap_enabled() const { return use_mmap_; }

private:
  static void put_string(FILE *outputfile, const string &strtmp);
  static void get_string(FILE *inputfile, string &strtmp);
  static int get_nsamples(const char *filename, int headersize, int nbits,
                          int nifs, int nchans);
  static long long sizeof_file(const char name[]);
  void reverse_channanl_data();

public:
  string filename;
  long int header_size;
  bool use_frequence_table;

  int telescope_id;
  int machine_id;
  int data_type;
  char rawdatafile[80];
  char source_name[80];
  int barycentric;
  int pulsarcentric;
  int ibeam;
  int nbeams;
  int npuls;
  int nbins;
  double az_start;
  double za_start;
  double src_raj;
  double src_dej;
  double tstart;
  double tsamp;
  int nbits;
  long int nsamples;
  int nifs;
  int nchans;
  double fch1;
  double foff;
  double refdm;
  double period;

  double *frequency_table;
  long int ndata;
  void *data;
  std::shared_ptr<void> data_owner;
  
  // Memory-mapped I/O
  MMapBuffer mmap_buffer_;
  bool use_mmap_ = true;  // Enable mmap by default

  FILE *fptr;
};

void get_telescope_name(int telescope_id, std::string &s_telescope);

#endif /* FILTERBANK_H_ */
