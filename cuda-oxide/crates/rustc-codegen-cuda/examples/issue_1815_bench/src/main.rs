/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Runtime cost of the #1815 loop shape: an 8-step accumulator loop inside a
//! runtime-length K loop, against the same work written out by hand.
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::wmma::mma_m16n8k16_f32_bf16;
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;
use std::time::Instant;

#[cuda_module]
mod kernels {
    use super::*;

    /// Natural form: `for n in 0..8` updating `acc[n]` (no annotation).
    #[kernel]
    pub fn mma_plain(a: &[u32], b: &[u32], ksteps: u32, mut out: DisjointSlice<f32>) {
        let lane = (thread::threadIdx_x() % 32) as usize;
        let mut acc = [[0f32; 4]; 8];
        let mut k = 0u32;
        while k < ksteps {
            let r = ((k % 64) as usize * 32 + lane) * 4;
            let av = [a[r], a[r + 1], a[r + 2], a[r + 3]];
            for n in 0..8 {
                let c = (n * 32 + lane) * 2;
                let bv = [b[c], b[c + 1]];
                acc[n] = unsafe { mma_m16n8k16_f32_bf16(acc[n], av, bv) };
            }
            k += 1;
        }
        let s = (acc[0][0] + acc[1][1])
            + (acc[2][2] + acc[3][3])
            + (acc[4][0] + acc[5][1])
            + (acc[6][2] + acc[7][3]);
        if let Some(o) = out.get_mut(thread::index_1d()) {
            *o = s;
        }
    }

    /// Reference: the same work with the 8 steps written out.
    #[kernel]
    pub fn mma_handunrolled(a: &[u32], b: &[u32], ksteps: u32, mut out: DisjointSlice<f32>) {
        let lane = (thread::threadIdx_x() % 32) as usize;
        let mut acc = [[0f32; 4]; 8];
        let mut k = 0u32;
        while k < ksteps {
            let r = ((k % 64) as usize * 32 + lane) * 4;
            let av = [a[r], a[r + 1], a[r + 2], a[r + 3]];
            macro_rules! step {
                ($n:literal) => {
                    let c = ($n * 32 + lane) * 2;
                    let bv = [b[c], b[c + 1]];
                    acc[$n] = unsafe { mma_m16n8k16_f32_bf16(acc[$n], av, bv) };
                };
            }
            step!(0);
            step!(1);
            step!(2);
            step!(3);
            step!(4);
            step!(5);
            step!(6);
            step!(7);
            k += 1;
        }
        let s = (acc[0][0] + acc[1][1])
            + (acc[2][2] + acc[3][3])
            + (acc[4][0] + acc[5][1])
            + (acc[6][2] + acc[7][3]);
        if let Some(o) = out.get_mut(thread::index_1d()) {
            *o = s;
        }
    }
}

fn main() {
    let ctx = CudaContext::new(0).expect("context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load module");
    let a: Vec<u32> = (0..64 * 32 * 4).map(|i: u32| 0x3f80_3f80 ^ (i & 0x00ff_00ff)).collect();
    let b: Vec<u32> = (0..8 * 32 * 2)
        .map(|i: u32| 0x3f80_3f80 ^ ((i * 7) & 0x00ff_00ff))
        .collect();
    let a_dev = DeviceBuffer::from_host(&stream, &a).unwrap();
    let b_dev = DeviceBuffer::from_host(&stream, &b).unwrap();
    // Enough blocks to fill a large GPU several times over.
    let blocks = 48 * 32u32;
    let threads = 128u32;
    let ksteps = 4096u32;
    let mut out = DeviceBuffer::<f32>::zeroed(&stream, (blocks * threads) as usize).unwrap();
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    let flop = 2.0 * 16.0 * 8.0 * 16.0 * 8.0 * ksteps as f64 * (blocks * threads / 32) as f64;
    let mut checks = Vec::new();
    for (name, which) in [("mma_plain", 0), ("mma_handunrolled", 1)] {
        let run = |out: &mut DeviceBuffer<f32>| {
            unsafe {
                if which == 0 {
                    module.mma_plain(&stream, cfg, &a_dev, &b_dev, ksteps, out)
                } else {
                    module.mma_handunrolled(&stream, cfg, &a_dev, &b_dev, ksteps, out)
                }
            }
            .unwrap()
        };
        for _ in 0..3 {
            run(&mut out);
        }
        stream.synchronize().unwrap();
        let mut best = f64::MAX;
        for _ in 0..5 {
            let t = Instant::now();
            for _ in 0..10 {
                run(&mut out);
            }
            stream.synchronize().unwrap();
            best = best.min(t.elapsed().as_secs_f64() * 1e3 / 10.0);
        }
        let host = out.to_host_vec(&stream).unwrap();
        checks.push(host);
        println!(
            "{name:18} {best:8.3} ms  {:7.1} TFLOP/s",
            flop / (best * 1e-3) / 1e12
        );
    }
    assert!(
        checks[0] == checks[1],
        "mma_plain and mma_handunrolled disagree"
    );
    println!("outputs identical: yes");
    println!("SUCCESS");
}
