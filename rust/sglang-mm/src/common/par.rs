//! The crate's only parallelism seam.

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Map `items`, short-circuiting on the first error. Output order matches input order.
#[cfg(feature = "parallel")]
pub fn try_map<'a, T, R, E>(
    items: &'a [T],
    f: impl Fn(&'a T) -> Result<R, E> + Send + Sync,
) -> Result<Vec<R>, E>
where
    T: Send + Sync,
    R: Send,
    E: Send,
{
    super::pool().install(|| items.par_iter().map(f).collect())
}

#[cfg(not(feature = "parallel"))]
pub fn try_map<'a, T, R, E>(
    items: &'a [T],
    f: impl Fn(&'a T) -> Result<R, E> + Send + Sync,
) -> Result<Vec<R>, E>
where
    T: Send + Sync,
    R: Send,
    E: Send,
{
    items.iter().map(f).collect()
}

/// Apply `f(chunk_index, chunk)` over disjoint `chunk_size`-element windows of `buf`.
#[cfg(feature = "parallel")]
pub fn for_chunks_mut<T: Send>(
    buf: &mut [T],
    chunk_size: usize,
    f: impl Fn(usize, &mut [T]) + Send + Sync,
) {
    super::pool().install(|| {
        buf.par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(index, chunk)| f(index, chunk));
    });
}

#[cfg(not(feature = "parallel"))]
pub fn for_chunks_mut<T: Send>(
    buf: &mut [T],
    chunk_size: usize,
    f: impl Fn(usize, &mut [T]) + Send + Sync,
) {
    for (index, chunk) in buf.chunks_mut(chunk_size).enumerate() {
        f(index, chunk);
    }
}

/// Run `f` with the CPU pool already entered.
#[cfg(feature = "parallel")]
pub fn in_pool<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    super::pool().install(f)
}

#[cfg(not(feature = "parallel"))]
pub fn in_pool<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    f()
}
