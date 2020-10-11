#pragma once

#include <memory>
//From Jared Hoberock
//http://github.com/jaredhoberock/personal

namespace variant {
namespace detail {

namespace uninit {

template<typename T> struct alignment_of_impl;

template<typename T, std::size_t size_diff>
  struct helper
{
  static const std::size_t value = size_diff;
};

template<typename T>
  struct helper<T,0>
{
  static const std::size_t value = alignment_of_impl<T>::value;
};

template<typename T>
  struct alignment_of_impl
{
  struct big { T x; char c; };

  static const std::size_t value = helper<big, sizeof(big) - sizeof(T)>::value;
};
  
} // end detail_uninit

template<typename T>
  struct alignment_of
    : uninit::alignment_of_impl<T>
{};

template<std::size_t Len, std::size_t Align>
  struct aligned_storage
{
  union type
  {
    unsigned char data[Len];
    struct __align__(Align) { } align;
  };
};

template<typename T>
  class uninitialized
{
  private:
    typename aligned_storage<sizeof(T), alignment_of<T>::value>::type storage;

    __host__ __device__ inline const T* ptr() const
    {
      return reinterpret_cast<const T*>(storage.data);
    }

    __host__ __device__ inline T* ptr()
    {
      return reinterpret_cast<T*>(storage.data);
    }

  public:
    // copy assignment
    __host__ __device__ inline uninitialized<T> &operator=(const T &other)
    {
      T& self = *this;
      self = other;
      return *this;
    }

    __host__ __device__ inline T& get()
    {
      return *ptr();
    }

    __host__ __device__ inline const T& get() const
    {
      return *ptr();
    }

    __host__ __device__ inline operator T& ()
    {
      return get();
    }

    __host__ __device__ inline operator const T&() const
    {
      return get();
    }

#pragma hd_warning_disable
    inline __host__ __device__ void construct()
    {
      ::new(ptr()) T();
    }

#pragma hd_warning_disable
    template<typename Arg>
    inline __host__ __device__ void construct(const Arg &a)
    {
      ::new(ptr()) T(a);
    }

#pragma hd_warning_disable
    inline __host__ __device__ void destroy()
    {
      T& self = *this;
      self.~T();
    }
};

}
}
