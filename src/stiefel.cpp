#include "stiefel.hpp"

stiefel::stiefel(unsigned n1, unsigned p1, unsigned r1) : manifold(n1, p1, r1) {
  Y = mat(n1, p1, fill::eye);
}

auto stiefel::evalHessian(vecMatIter amb_grad, vecMatIter amb_hess,
                          tangent_vector &Z_tv) -> tangent_vector {
  tangent_vector hess;
  // project amb_hess to the tangent space
  hess = evalGradient(amb_hess);
  arma::mat xi_normal = evalNormal(*amb_grad);
  hess = hess + WeingartenMap(xi_normal, Z_tv);
  return hess;
}

auto stiefel::WeingartenMap(const arma::mat &xi_normal,
                            const tangent_vector &Z_tv) -> tangent_vector {
  tangent_vector Weingarten;
  Weingarten = create_tangentV_at_current_point();
  arma::mat YU = Z_tv(0).t() * xi_normal;
  Weingarten(0) = -Z_tv(0) * Y.t() * xi_normal;
  Weingarten(0) -= 0.5 * Y * (YU + YU.t());
  return Weingarten;
}

//! Compute gradient from the ambient space.
/*!l
amb_grad (G) is the ambient gradient.
G - Y* Sym(Y^TG) and
Sym(X) = 0.5(X + X^T)
*/
auto stiefel::evalGradient(vecMatIter amb_grad) -> tangent_vector {
  tangent_vector grad;
  grad = create_tangentV_at_current_point();
  grad("xi") = (*amb_grad) - evalNormal(*amb_grad);
  return grad;
}

arma::mat stiefel::evalNormal(const arma::mat &ambM) {
  mat xi_normal;
  xi_normal = Y.t() * ambM;

  xi_normal = 0.5 * Y * (xi_normal + xi_normal.t());
  return xi_normal;
}

auto stiefel::retraction(tangent_vector &xi) -> manifold_ptr {
  // xi.ownership_check(id_code);

  manifold_ptr result;
  result = retract_QR(xi);

  return result;
}

auto stiefel::retract_QR(tangent_vector &xi) -> manifold_ptr {
  shared_ptr<stiefel> result = make_shared<stiefel>(n, p, r);
  arma::mat Yt = Y + xi(0);
  arma::mat retract_Q, retract_R; // gradient in the tangent space

  arma::qr_econ(retract_Q, retract_R, Yt);
  // if (retract_R(0, 0) < 0) {
  //   retract_Q = -retract_Q;
  // }
  
  // We need to make sure the diagonal elements
  // of retract_R are all positive
  int i;
  for(i = 0; i < retract_R.n_cols; i++)
    retract_Q.col(i) *= (retract_R(i,i) > 0) - (retract_R(i,i) < 0);

  Yt = retract_Q;
  result->Y = Yt;

  return result;
}

void stiefel::initial_point(vecMatIter rObj) {
  arma::mat retract_U, retract_V;
  arma::qr_econ(retract_U, retract_V, *rObj);
  Y = retract_U;
}

// second argument unused
//     }else if(retraction==2){//Cayley
//         //Rcpp::Rcout<<"Cayley"<<endl;
//         // enve if we don't change metric, this retraction still works
//         arma::mat A=Y.t()*descD;  //another approach that
//         descD=YA+(Y_annhilator)*B;
//         // Looking for z=1/2YA+(Y_annhilator)*B;
//         arma::mat z=descD-1/2*Y*A;
//         // Then Omega=z*Y.t()-Y*z.t()
//         arma::mat Omega=z*Y.t()-Y*z.t();
//         Yt=arma::eye(n,n)-stepSize/2*Omega;
//         Yt=Yt.i()*(arma::eye(n,n)+stepSize/2*Omega)*Y;
//         //  The following def cannot result in correct answers;
//         //    arma::mat Omega=-Y.t()*descD;
//         //    arma::mat temp=arma::eye(p,p)-stepSize/2*Omega;
//         //    Yt=Y*temp.i()*(arma::eye(p,p)+stepSize/2*Omega);
//     }
//     return Yt;
// }

//
//
//
// arma::mat stiefel::genretract(double stepSize, const arma::mat &Z){
//  if(retraction==1){//QR retraction
//    Yt=Y+stepSize*Z;
//    arma::qr_econ(retract_Q,retract_R,Yt);
//    if(retract_R(0,0)<0){
//      retract_Q=-retract_Q;
//    }
//    Yt=retract_Q;
//  }else if(retraction==2){//cayley
//    arma::mat A=Y.t()*Z;  //another approach that z=YA+Y_annhilator*B;
//    arma::mat Omega=(Z-1/2*Y*A)*Y.t()-Y*(Z.t()-1/2*A.t()*Y.t());
//    Yt=arma::eye(n,n)-stepSize/2*Omega;
//    Yt=Yt.i()*(arma::eye(n,n)+stepSize/2*Omega)*Y;
//  }
//  return Yt;
//}

//
// void stiefel::set_particle(){
//     arma::mat y_temp=arma::randn(n,p);
//     arma::mat Q,R;
//     arma::qr_econ(Q,R,y_temp);
//     Y=Q;
//     arma::mat velocity_temp=arma::randn(n,p);
//     //psedo gradient as velocity;
//     evalGradient(velocity_temp,"particleSwarm");
// }
