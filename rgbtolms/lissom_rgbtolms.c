// C Function to use rgbtolms.h for rgbtolms function.
// Preparation and testing for inclusion in LISSOM 5.0
// by Judah De Paula
// December 5, 2005
#include <stdio.h>
#include "rgbtolms.h"

// Pre: 0.0 <= r,g,b <= 1.0;  lms: 3 element float array for result.
// Post: lms[0] = l, lms[1] = m, lms[2] = s;  0.0 <= l,m,s <= 1.0.  
void rgbtolms(float r, float g, float b, float lms[3]) {
  int i;
  float l,m,s,gamma_r,gamma_g,gamma_b;
  float emission_fcn[WAVELENGTH_POINTS];

  l = 0;  m = 0;  s = 0;
  gamma_r = gamma_fcn[(int) (r*255.0)];
  gamma_g = gamma_fcn[(int) (g*255.0)];
  gamma_b = gamma_fcn[(int) (b*255.0)];
  for (i=0;i<WAVELENGTH_POINTS;i++) {
    emission_fcn[i] = gamma_r*rgb_fcn[0][i] + gamma_g*rgb_fcn[1][i] + gamma_b*rgb_fcn[2][i];
  }
  for (i=0;i<WAVELENGTH_POINTS;i++) {
    l += emission_fcn[i] * lms_fcn[0][i];
    m += emission_fcn[i] * lms_fcn[1][i];
    s += emission_fcn[i] * lms_fcn[2][i];
  }
  lms[0] = l;
  lms[1] = m;
  lms[2] = s;
}


void test_rgbtolms(float r, float g, float b, float l, float m, float s) {
  float lms[3], eps = 0.0001;
  rgbtolms(r,g,b,lms);
  if (fabs(lms[0]-l) > eps || fabs(lms[1]-m) > eps || fabs(lms[2]-s) > eps) {
    printf("Test failure: RGB: %f %f %f did not generate %f %f %f\n",r,g,b,l,m,s);
    printf("Generated: %f %f %f\n", lms[0],lms[1],lms[2]);
  }
}  


//  A few tests to make sure the program is creating correct values:
//  These code examples from rgbtolms Python script:
//    assert RGB_to_LMS((0,0,0)) == (0,0,0), "Output not matched"
//    assert RGB_to_LMS((85,85,85)) == (23,23,23), "Output not matched"
//    assert RGB_to_LMS((128,128,128)) == (56,56,56), "Output not matched"
//    assert RGB_to_LMS((170,170,170)) == (105,105,105), "Output not matched"
//    assert RGB_to_LMS((255,255,255)) == (255,255,255), "Output not matched"
//    assert RGB_to_LMS((255,0,0)) == (73,29,4), "Output not matched"
//    assert RGB_to_LMS((0,255,0)) == (140,159,14), "Output not matched"
//    assert RGB_to_LMS((0,0,255)) == (42,67,237), "Output not matched"
void run_tests(void) {
  test_rgbtolms(0,0,0,             0,0,0);
  test_rgbtolms(1.0,1.0,1.0,       1.0,1.0,1.0);
  test_rgbtolms(1.0,0,0,           0.287745,0.11398,0.015629);
  test_rgbtolms(0,1.0,0,           0.547419,0.623551,0.055946);
  test_rgbtolms(0,0,1.0,           0.164830,0.262465,0.928447);
  test_rgbtolms(0.333,0.333,0.333, 0.08690,0.08690,0.08690);
  test_rgbtolms(0.5,0.5,0.5,       0.21576,0.21576,0.21576);
  test_rgbtolms(0.666,0.666,0.666, 0.40453,0.40453,0.40453);
}

int main(void){
  run_tests();
}
