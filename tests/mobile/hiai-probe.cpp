#include "hiai_runtime.hpp"
#include <cmath>
#include <cstdio>
#include <chrono>
#include <algorithm>
#include <cstdlib>
int main(int argc,char **argv){
 int k=argc>1?atoi(argv[1]):32,n=argc>2?atoi(argv[2]):32,m=argc>3?atoi(argv[3]):16;
 if(k<1||n<1||m<1||k>16384||n>16384||m>16384)return 2;
 std::vector<float>w(size_t(k)*n),x(size_t(k)*m),y(size_t(n)*m);
 for(size_t i=0;i<w.size();i++)w[i]=std::sin(float(i)*.17f)*.1f;
 for(size_t i=0;i<x.size();i++)x[i]=std::cos(float(i)*.13f)*.1f;
 ed_hiai::Conv c(k,n,m,w.data());
 if(!c.valid()){fprintf(stderr,"HIAI_UNAVAILABLE %s\n",c.error.c_str());return 3;}
 auto start=std::chrono::steady_clock::now();
 if(!c.run(x.data(),y.data())){fprintf(stderr,"HIAI_EXEC_FAILED %s\n",c.error.c_str());return 4;}
 double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
 double sq=0,maxe=0,refmax=0;
 for(int j=0;j<m;j++)for(int i=0;i<n;i++){
  double ref=0;for(int t=0;t<k;t++)ref+=double(w[size_t(i)*k+t])*x[size_t(j)*k+t];
  double e=y[size_t(j)*n+i]-ref;sq+=e*e;maxe=std::max(maxe,std::abs(e));refmax=std::max(refmax,std::abs(ref));
 }
 double rmse=std::sqrt(sq/(n*m));
 printf("{\"backend\":\"hiai-single-op\",\"k\":%d,\"n\":%d,\"m\":%d,\"precheck\":%d,\"workspace_bytes\":%zu,\"execute_ms\":%.4f,\"rmse\":%.8g,\"max_error\":%.8g,\"reference_max\":%.8g}\n",k,n,m,c.support,c.workspace_bytes,ms,rmse,maxe,refmax);
 return std::isfinite(rmse)&&maxe<.01+refmax*.01?0:5;
}
