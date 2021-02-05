use nalgebra::base::allocator::Allocator;
use nalgebra::{DefaultAllocator, DimName, VectorN};
use std::ops::{Add, Index, Mul, Sub};

pub trait OneStep<Sys: HybridSystem> {
    fn step(&self, system: &Sys, t: f64, h: f64) -> (Sys::State, f64);
    fn hmin(&self) -> f64;
    fn hmax(&self) -> f64;
}

pub trait HybridSystem
where
    Self::State: 'static + State,
{
    type State;

    fn time(&self) -> f64;
    fn state(&self) -> &Self::State;
    fn start(&mut self, time: f64, state: Self::State);
    fn detect(&self, time: f64, state: &Self::State) -> bool;
    fn submit(&mut self, time: f64, state: Self::State);
    fn linearize(&self, time: f64, state: &Self::State) -> Self::State;
}

pub struct RungeKutta {
    hmin: f64,
    hmax: f64,
}

impl<Sys: HybridSystem> OneStep<Sys> for RungeKutta {
    fn step(&self, system: &Sys, t: f64, h: f64) -> (Sys::State, f64) {
        let x0 = system.state();
        let h2 = h / 2.0;
        let k1 = system.linearize(t, x0);
        let k2 = system.linearize(t + h2, &(x0.clone() + k1.clone() * h2));
        let k3 = system.linearize(t + h2, &(x0.clone() + k2.clone() * h2));
        let k4 = system.linearize(t + h2, &(x0.clone() + k3.clone() * h));
        (x0.clone() + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (h / 6.0), h)
    }

    fn hmin(&self) -> f64 {
        self.hmin
    }

    fn hmax(&self) -> f64 {
        self.hmax
    }
}

pub struct Euler {
    hmin: f64,
    hmax: f64,
}

impl<Sys: HybridSystem> OneStep<Sys> for Euler {
    fn step(&self, system: &Sys, t: f64, h: f64) -> (Sys::State, f64) {
        let x0 = system.state();
        let k0 = system.linearize(t, x0);
        let x1 = x0.clone() + k0.clone() * h;
        let h_0_5 = 0.5 * h;
        let x_0_5: Sys::State = x0.clone() + k0 * h_0_5;
        let k_0_5 = system.linearize(t + h_0_5, &x_0_5);
        let x1_d: Sys::State = x_0_5.clone() + k_0_5 * h_0_5;

        let err: f64 = (x1_d.clone() - x1).norm();
        let new_h = (0.05 / (2.0 * err)).sqrt();
        let mut h = 0.9 * h * f64::min(f64::max(new_h, 0.3), 2.0);
        h = f64::min(h, self.hmax);
        h = f64::max(h, self.hmin);

        (x1_d, h)
    }

    fn hmin(&self) -> f64 {
        self.hmin
    }

    fn hmax(&self) -> f64 {
        self.hmax
    }
}

pub struct Ode23 {
    pub hmin: f64,
    pub hmax: f64,
    pub reltol: f64,
    pub abstol: f64,
}

// implementation acc:
// https://blogs.mathworks.com/cleve/2014/05/26/ordinary-differential-equation-solvers-ode23-and-ode45/
impl<Sys: HybridSystem> OneStep<Sys> for Ode23 {
    fn step(&self, system: &Sys, t: f64, mut h: f64) -> (Sys::State, f64) {
        loop {
            let x0 = system.state();
            let k1 = system.linearize(t, x0);
            let k2 = system.linearize(t + 0.5 * h, &(x0.clone() + k1.clone() * 0.5 * h));
            let k3 = system.linearize(t + 0.75 * h, &(x0.clone() + k2.clone() * 0.75 * h));
            let t1 = t + h;
            let x1 =
                x0.clone() + (k1.clone() * 2.0 + k2.clone() * 3.0 + k3.clone() * 4.0) * (h / 9.0);
            let k4 = system.linearize(t1, &x1);
            let e1 = ((k1 * -5.0 + k2 * 6.0 + k3 * 8.0 - k4 * 9.0) * (h / 72.0)).abs();
            let mut max_ratio = 1e-6;
            for k in 0_usize..x0.len() {
                let bound = (self.reltol * x0[k]).abs().max(self.abstol);
                let ratio = e1[k] / bound;
                if ratio > max_ratio {
                    max_ratio = ratio;
                }
            }
            h *= (1.0 / (1.2 * max_ratio)).min(3.0).max(0.3);
            h = h.min(self.hmax);
            h = h.max(self.hmin);
            if max_ratio < 1.0 {
                return (x1, h.min(self.hmax).max(self.hmin));
            }
            if h <= self.hmin {
                return (x1, self.hmin);
            }
        }
    }

    fn hmin(&self) -> f64 {
        self.hmin
    }

    fn hmax(&self) -> f64 {
        self.hmax
    }
}

pub trait State:
    Sized
    + Clone
    + Mul<f64, Output = Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Index<usize, Output = f64>
{
    fn norm(&self) -> f64;
    fn abs(&self) -> Self;
    fn len(&self) -> usize;
}

impl<D: DimName> State for VectorN<f64, D>
where
    DefaultAllocator: Allocator<f64, D>,
{
    fn norm(&self) -> f64 {
        self.iter().fold(0.0, |x, y| x + y * y).sqrt()
    }

    fn abs(&self) -> Self {
        self.abs()
    }

    fn len(&self) -> usize {
        self.len()
    }
}

pub fn simulate_to_transition<Sys: HybridSystem, Stepper: OneStep<Sys>>(
    sys: &mut Sys,
    stepper: &mut Stepper,
    tend: f64,
) -> f64 {
    let mut t = sys.time();
    sys.start(t, sys.state().clone());
    let mut h = stepper.hmin();
    loop {
        let (newx, hnew) = stepper.step(sys, t, h);
        let sys_change = sys.detect(t + h, &newx);
        if sys_change || t > tend {
            if h < stepper.hmin() {
                t += stepper.hmin();
                sys.submit(t, newx);
                return t;
            }
            h /= 4.0;
        } else {
            t += h;
            h = hnew;
            sys.submit(t, newx);
        }
    }
}

pub fn simulate<Sys: HybridSystem, Stepper: OneStep<Sys>>(
    sys: &mut Sys,
    stepper: &mut Stepper,
    tend: f64,
) -> f64 {
    while sys.time() < tend {
        simulate_to_transition(sys, stepper, tend);
    }
    sys.time()
}
