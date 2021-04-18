use colored_noise::PerlinNoise;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use itertools::Itertools;
use ndarray::prelude::*;

fn perlin2d_bench(c: &mut Criterion) {
    c.bench_function("perlin noise 2d", |b| {
        b.iter(|| {
            let x = PerlinNoise::new(2000000, 8).gen2d(200, 200);
            let x = &x + 0.5 * &PerlinNoise::new(20000, 16).gen2d(200, 200);
            let x = &x + 0.25 * &PerlinNoise::new(2000, 32).gen2d(200, 200);
            let x = &x + 0.125 * &PerlinNoise::new(200, 64).gen2d(200, 200);
            let x = &x + 0.0625 * &PerlinNoise::new(20, 128).gen2d(200, 200);
        })
    });
}

criterion_group!(benches, perlin2d_bench);
criterion_main!(benches);
