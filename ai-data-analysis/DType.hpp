#pragma once
enum class DType : uint8_t { Float32, Float64, Int32, UInt8 };

template <typename T>
constexpr DType dtype_of() {
    if constexpr (std::is_same_v<T, float>)    return DType::Float32;
    else if constexpr (std::is_same_v<T, double>)   return DType::Float64;
    else if constexpr (std::is_same_v<T, int32_t>)  return DType::Int32;
    else if constexpr (std::is_same_v<T, uint8_t>)  return DType::UInt8;
    else static_assert(sizeof(T) == 0, "Unsupported DType");
}