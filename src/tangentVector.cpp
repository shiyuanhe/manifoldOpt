#include "tangentVector.hpp"

auto tangent_vector::operator+(const tangent_vector &rhs) const
    -> tangent_vector {
  tangent_vector result(id_code);
  result.set_nComponent(nComponent);
  for (size_type i = 0; i < nComponent; i++)
    result.tangent_components.at(i) =
        tangent_components.at(i) + rhs.tangent_components.at(i);
  return result;
}

auto tangent_vector::operator-(const tangent_vector &rhs) const
    -> tangent_vector {
  tangent_vector result(id_code);
  result.set_nComponent(nComponent);
  for (size_type i = 0; i < nComponent; i++)
    result.tangent_components.at(i) =
        tangent_components.at(i) - rhs.tangent_components.at(i);
  return result;
}

auto tangent_vector::operator-() const -> tangent_vector {
  tangent_vector result(id_code);
  result.set_nComponent(nComponent);
  for (size_type i = 0; i < nComponent; i++)
    result.tangent_components.at(i) = -tangent_components.at(i);
  return result;
}

auto tangent_vector::operator*(const double scalar) const -> tangent_vector {
  tangent_vector result(id_code);
  result.set_nComponent(nComponent);
  for (size_type i = 0; i < nComponent; i++)
    result.tangent_components.at(i) = tangent_components.at(i) * scalar;
  return result;
}

// Merge 2 product components
auto tangent_vector::operator*(const tangent_vector &rhs) const
    -> tangent_vector {
  tangent_vector result(id_code);
  result = *this;
  result.tangent_components.insert(result.tangent_components.end(),
                                   rhs.tangent_components.begin(),
                                   rhs.tangent_components.end());
  result.nComponent = result.tangent_components.size();
  return result;
}
