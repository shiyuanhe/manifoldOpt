#ifndef TANGENT_VECTOR_CLASS
#define TANGENT_VECTOR_CLASS

#include "tangentVectorBase.hpp"

class tangent_vector {
public:
  // Initialize
  tangent_vector(const string assign_id = "") {
    tangent_components.resize(1);
    nComponent = 1;
    id_code = assign_id;
    currentViewK = 0;
  }

  // vector operation
  auto operator+(const tangent_vector &rhs) const -> tangent_vector;
  auto operator-(const tangent_vector &rhs) const -> tangent_vector;
  auto operator-() const -> tangent_vector;
  auto operator*(const double scalar) const -> tangent_vector;
  auto operator*(const tangent_vector &rhs) const -> tangent_vector;

  // view component
  void set_nComponent(size_type size_) {
    nComponent = size_;
    tangent_components.clear();
    tangent_components.resize(nComponent);
  }

  void set_currentViewK(const size_type currentViewK_) {
    if (currentViewK_ >= nComponent)
      throw runtime_error("Component out of bound!");
    currentViewK = currentViewK_;
  }

  // modify components
  void resize(size_type size_) {
    tangent_components.at(currentViewK).resize(size_);
  }

  void assign_names(const vector<string> &str_) {
    tangent_components.at(currentViewK).assign_names(str_);
  }

  // Fetch element
  arma::mat &operator()(const size_type int_index = 0) {
    return tangent_components.at(currentViewK)(int_index);
  }

  const arma::mat &operator()(const size_type int_index = 0) const {
    return tangent_components.at(currentViewK)(int_index);
  }

  arma::mat &operator()(const string str_index) {
    return tangent_components.at(currentViewK)(str_index);
  }

  const arma::mat &operator()(const string str_index) const {
    return tangent_components.at(currentViewK)(str_index);
  }

  void ownership_check(const string owner_id) const {
    if (id_code != owner_id)
      throw runtime_error(
          "The tangent vector does not belong to the tangent space!");
  }

  void ownership_set(const string owner_id) { id_code = owner_id; }

private:
  string id_code;
  // size_type currentViewK;
  size_type nComponent;
  vector<tangent_vector_base> tangent_components;
  size_type currentViewK;
};

#endif
