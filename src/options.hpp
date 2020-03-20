#ifndef R_OPTION_CLASS
#define R_OPTION_CLASS


#include "manifold_include.hpp"


class rOptions{
public:
    
    void Rprint_options(){
        for(it = opts.begin(); it!=opts.end(); it++)
            Rcpp::Rcout<<it->first<<" = "<<it->second<<std::endl;
    }
    
    void Rset_options(Rcpp::List rList){
        std::vector< std::string> params_names = rList.names();
        int n = params_names.size();
        for(int i=0; i<n; i++){
            it = opts.find(params_names[i]);
            if(it==opts.end())
                printToR_NameNotFound(params_names[i]);
            else
                it->second = Rcpp::as<double>(rList[i]);
        }
    }
    
    double& operator[] (string sIndex) {
        return opts.at(sIndex);
    }
    
    void insert(string sIndex, double val){
        std::pair<string, double> valPair;
        valPair.first = sIndex;
        valPair.second = val;
        opts.insert(valPair);
    }
    
    void clear(){
        opts.clear();
    }
    
private:
    map<string, double> opts;
    typename map<string, double>::iterator it;
    
    void printToR_NameNotFound(string cName){
        Rcpp::Rcerr<<"No option named '"<<
            cName<<"'"<<std::endl;
    }
};


#endif
