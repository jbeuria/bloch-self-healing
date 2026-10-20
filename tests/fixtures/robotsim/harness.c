/* Audit-only vector compatibility layer and circular obstacle adapter.
 * The included interaction function bodies are unchanged upstream code. */
#include <math.h>
#include <stdlib.h>
#define MAX(a,b) (((a)>(b))?(a):(b))
typedef struct { int NumberOfAgents; double **Coordinates; double **Velocities; } phase_t;
typedef struct { double x,y,r; } obstacle_t;
void NullVect(double *o,int n){for(int i=0;i<n;i++)o[i]=0;}
double VectAbs(double *a){return sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]);}
void VectDifference(double *o,double *a,double *b){for(int i=0;i<3;i++)o[i]=a[i]-b[i];}
void VectSum(double *o,double *a,double *b){for(int i=0;i<3;i++)o[i]=a[i]+b[i];}
void UnitVect(double *o,double *a){double r=VectAbs(a);for(int i=0;i<3;i++)o[i]=r>1e-12?a[i]/r:0;}
void MultiplicateWithScalar(double *o,double *a,double s,int n){for(int i=0;i<n;i++)o[i]=a[i]*s;}
double DistanceOfNearestPointOfObstacle(double *q,obstacle_t *o,double *x){
 double dx=x[0]-o->x,dy=x[1]-o->y,r=hypot(dx,dy);
 q[0]=o->x+o->r*dx/r;q[1]=o->y+o->r*dy/r;q[2]=0;return r-o->r;
}
#include "source_functions.inc"
void audit_friction(int n,double *x,double *v,double cf,double slack,double a,double p,double r0,double *out){
 double **xx=malloc(n*sizeof(double*)),**vv=malloc(n*sizeof(double*));
 for(int i=0;i<n;i++){xx[i]=x+3*i;vv[i]=v+3*i;}
 phase_t ph={n,xx,vv};for(int i=0;i<n;i++)FrictionLinSqrt(out+3*i,&ph,cf,slack,a,p,r0,i,2);
 free(xx);free(vv);
}
void audit_shill(int n,double *x,double *v,double *disk,double speed,double r0,double a,double p,double *out){
 double **xx=malloc(n*sizeof(double*)),**vv=malloc(n*sizeof(double*));
 for(int i=0;i<n;i++){xx[i]=x+3*i;vv[i]=v+3*i;NullVect(out+3*i,3);}
 phase_t ph={n,xx,vv};obstacle_t ob={disk[0],disk[1],disk[2]};
 for(int i=0;i<n;i++)Shill_Obstacle_LinSqrt(out+3*i,&ph,&ob,speed,r0,a,p,i);
 free(xx);free(vv);
}
