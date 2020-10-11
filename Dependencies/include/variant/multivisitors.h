#include <variant/variant.h>

namespace variant {

namespace detail {

template<typename Fn,
         typename V0,
         typename V1>
struct bivisitor {
   
    Fn fn;
    const V1& v1;

    using O = decltype(fn(std::declval<typename V0::cons_type::head_type>(),
                          std::declval<typename V1::cons_type::head_type>()));
    
    template<typename T0>
    struct monovisitor {
        
        Fn fn;
        const T0& t0;

        template<typename T1>
        __host__ __device__
        O operator()(const T1& t1) const {
            return fn(t0, t1);
        }

    };

    template<typename T0>
    __host__ __device__
    O operator()(const T0& t0) const {
        return apply_visitor(monovisitor<T0>{fn, t0},
                             v1);
    }

};

}


//Apply visitor to variant
template<typename Fn, typename V0, typename V1>
__host__ __device__
auto apply_visitor(Fn fn, const V0& v0, const V1& v1) -> 
    decltype(fn(std::declval<typename V0::cons_type::head_type>(),
                std::declval<typename V1::cons_type::head_type>()))
{
    return apply_visitor(detail::bivisitor<Fn, V0, V1>{fn, v1},
                         v0);
}

}
