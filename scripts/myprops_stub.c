/*
 * MyProps stub for ARM64 - wraps CoolProp's public C API
 * Only implements functions used by Steps.Utilities.CoolProp.*
 * PCHE and complex struct functions are stubs (not used by our cycle).
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* Forward-declare CoolProp's C API (from CoolPropLib.h) */
extern double PropsSI(const char *Output, const char *Name1, double Prop1,
                      const char *Name2, double Prop2, const char *Ref);

/* ── Core function used by Steps.Utilities.CoolProp.PropsSI ────────── */
double MyPropsSI(const char *Output, const char *Name1, double Prop1,
                 const char *Name2, double Prop2, const char *Ref)
{
    return PropsSI(Output, Name1, Prop1, Name2, Prop2, Ref);
}

/* ── Batch helper used by Steps.Utilities.CoolProp.MyPropsSI ─────────
 * Signature (from CoolProp.mo external declaration):
 *   external "C" T = MyPropsSI_pH(p, H, fluidName, mu, k, rho);
 * Returns T; mu, k, rho are output arguments.
 */
double MyPropsSI_pH(double p, double H, const char *FluidName,
                    double *mu, double *k, double *rho)
{
    double T   = PropsSI("T",   "P", p, "H", H, FluidName);
    *mu        = PropsSI("V",   "P", p, "H", H, FluidName);
    *k         = PropsSI("L",   "P", p, "H", H, FluidName);
    *rho       = PropsSI("D",   "P", p, "H", H, FluidName);
    return T;
}

/* ── Batch helper: MyPropsSI_pT ─────────────────────────────────────── */
void MyPropsSI_pT(double p, double T, const char *FluidName,
                  double *h, double *rho)
{
    *h   = PropsSI("H", "P", p, "T", T, FluidName);
    *rho = PropsSI("D", "P", p, "T", T, FluidName);
}

/* ── Utility stubs ───────────────────────────────────────────────────── */
double from_deg(double deg)  { return deg * (3.14159265358979323846 / 180.0); }
double from_bar(double p_bar){ return p_bar * 1e5; }
double from_degC(double degC){ return degC + 273.15; }

double material_conductivity(double T, int extrapolate)
{
    (void)T; (void)extrapolate;
    return 16.0;   /* 316 SS at ~600 K, W/(m·K) */
}

/* ── PCHE stubs (not used by simple recuperated cycle) ──────────────── */
double PCHE_OFFD_Simulation(const void *name, const void *mh, const void *mc,
                            void *geo, void *cor, void *sim, void *bc,
                            void *retOutput)
{
    (void)name; (void)mh; (void)mc; (void)geo; (void)cor;
    (void)sim;  (void)bc; (void)retOutput;
    return 0.0;
}

double PCHE_OFFD_Simulation_UQ_out(const void *name, const void *mh,
                                   const void *mc, void *geo, void *cor,
                                   void *sim, void *bc, void *retOutput,
                                   double *Q, double *U)
{
    (void)name; (void)mh; (void)mc; (void)geo; (void)cor;
    (void)sim;  (void)bc; (void)retOutput;
    if (Q) *Q = 0.0;
    if (U) *U = 0.0;
    return 0.0;
}

void *NewThermoState_pT(double p, double T, double mdot, const char *medium)
{
    (void)p; (void)T; (void)mdot; (void)medium;
    return NULL;
}

int the_same(double x, double y, double eps, double *diff_per)
{
    double d = fabs(x - y) / (fabs(x) > 1e-12 ? fabs(x) : 1e-12);
    if (diff_per) *diff_per = d;
    return d <= eps;
}

double print_path_state(const char *name, const char *media, void *st, int lv)
{
    (void)name; (void)media; (void)st; (void)lv;
    return 0.0;
}

void test_struct_param(void *sp, void *geo, void *bc,
                       double *hh, double *hc, double *ph, double *pc,
                       size_t N)
{
    (void)sp; (void)geo; (void)bc; (void)hh; (void)hc;
    (void)ph; (void)pc; (void)N;
}

void setState_C_impl(double p, double M, void *state)
{
    (void)p; (void)M; (void)state;
}

/* ── CoolProp low-level wrapper stubs ───────────────────────────────── */
void *init_cp_wrapper(const char *medium, const char *name)
{
    (void)medium; (void)name;
    return NULL;
}

double cp_query(void *wrapper, const char *input_pair,
                double val1, double val2, const char *output_name)
{
    (void)wrapper; (void)input_pair; (void)val1;
    (void)val2;    (void)output_name;
    return 0.0;
}

void close_cp_wrapper(void *state) { (void)state; }
