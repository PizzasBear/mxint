use std::time::{Duration, Instant};

use rand::prelude::*;

pub mod big_int;
mod u64_ext;

use big_int::BigUint;
use u64_ext::U64Ext;

const ROUNDS: usize = 1;
const DIGITS: usize = 1000000;

fn main() {
    let mut rng = SmallRng::from_os_rng();

    let mut dur = Duration::ZERO;
    for _ in 0..ROUNDS {
        let num1 = BigUint::from_digits((&mut rng).random_iter().take(DIGITS).collect());
        let num2 = BigUint::from_digits((&mut rng).random_iter().take(DIGITS).collect());

        let start = Instant::now();
        num2.mul(&num1);
        dur += start.elapsed();
    }

    println!("DURATION: {dur:?}");
}

// #[derive(Clone)]
// struct BigUint {
//     zx_digits: Box<[u64]>,
//     len: usize,
// }
//
// impl BigUint {
//     fn digits(&self) -> &[u64] {
//         &self.zx_digits[..self.len]
//     }
//     fn digits_mut(&mut self) -> &mut [u64] {
//         &mut self.zx_digits[..self.len]
//     }
//     fn bit_len(&self) -> usize {
//         match self.len.checked_sub(1) {
//             Some(i) => 64 * self.len - self.zx_digits[i].leading_zeros() as usize,
//             None => 0,
//         }
//     }
//     fn mul(&self, other: &Self) -> Self {
//         let mut digits = vec![0; self.len + other.len].into_boxed_slice();
//         big_int::karatsuba_mul(&mut digits, self.digits(), other.digits());
//         Self::from_digits(digits)
//     }
//
//     fn from_digits(digits: Box<[u64]>) -> Self {
//         let mut slf = Self {
//             len: digits.len(),
//             zx_digits: digits,
//         };
//         slf.trim_zeros();
//         slf
//     }
//
//     fn trim_zeros(&mut self) {
//         while let Some(last) = self.len.checked_sub(1) {
//             match self.zx_digits[last] {
//                 0 => self.len = last,
//                 _ => break,
//             }
//         }
//     }
//
//     pub fn div_small(&mut self, rhs: u64) -> u64 {
//         let digits = self.digits_mut();
//
//         let mut carry = 0;
//         for digit in digits.iter_mut().rev() {
//             let total = (carry as u128) << 64 | *digit as u128;
//             *digit = (total / rhs as u128) as _;
//             carry = (total % rhs as u128) as _;
//         }
//
//         self.trim_zeros();
//
//         carry
//     }
// }
//
// impl fmt::Display for BigUint {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         use std::fmt::Write;
//
//         if 256 < self.digits().len() {
//             write!(f, "<uint with {} bits>", self.bit_len())?;
//             return Ok(());
//         } else if self.digits().len() == 0 {
//             f.write_char('0')?;
//             return Ok(());
//         }
//
//         let mut decimal_digits = String::with_capacity(self.bit_len());
//
//         let mut num = self.clone();
//
//         while num.digits().len() != 0 {
//             let start = decimal_digits.len();
//             write!(
//                 decimal_digits,
//                 "{:0>16}",
//                 num.div_small(10_000_000_000_000_000),
//             )?;
//             unsafe { decimal_digits.as_bytes_mut()[start..].reverse() };
//         }
//
//         unsafe { decimal_digits.as_bytes_mut().reverse() };
//
//         f.write_str(decimal_digits.trim_start_matches('0'))
//     }
// }
//
// fn main() {
//     let mut n = BigUint::from_digits(Box::new([10_000000_000000_000000]));
//     for i in 0..15 {
//         println!("i={i}: num_digits={}", n.digits().len());
//         n = n.mul(&n);
//         println!("  n={n}");
//     }
// }
//
