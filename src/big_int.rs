use std::{cmp::Ordering, fmt, mem, ops};

use crate::U64Ext;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BigUint {
    digits: Vec<u64>,
}

impl BigUint {
    const ZERO: Self = Self { digits: Vec::new() };
    pub const fn zero() -> Self {
        Self::ZERO
    }
    pub fn with_capacity(num_digits: usize) -> Self {
        Self {
            digits: Vec::with_capacity(num_digits),
        }
    }
    pub fn from_u64(value: u64) -> Self {
        if value == 0 {
            return Self::ZERO;
        }
        Self {
            digits: vec![value],
        }
    }
    pub fn from_digits(digits: Vec<u64>) -> Self {
        let mut slf = Self { digits };
        slf.trim_zeros();
        slf
    }
    pub fn into_digits(self) -> Vec<u64> {
        self.digits
    }
    pub fn is_zero(&self) -> bool {
        self.digits.is_empty()
    }
    pub fn num_digits(&self) -> usize {
        self.digits.len()
    }
    pub fn bit_len(&self) -> u64 {
        match self.digits.last() {
            Some(&last_digit) => {
                ((self.digits.len() as u64).checked_mul(64))
                    .expect("Bit length must be representable by a `usize`")
                    - last_digit.leading_zeros() as u64
            }
            None => 0,
        }
    }
    pub fn digits(&self) -> &[u64] {
        &self.digits
    }
    fn trim_zeros(&mut self) {
        while let Some(&0) = self.digits.last() {
            self.digits.pop();
        }
    }
    fn zero_extend(&mut self, num_digits: usize) {
        self.digits.resize(self.digits.len().max(num_digits), 0);
    }
    fn clone_with_capacity(&self, num_digits: usize) -> Self {
        let mut clone = Self::with_capacity(num_digits.max(self.num_digits()));
        clone.digits.extend_from_slice(&self.digits);
        clone
    }
    pub fn add_inplace(&mut self, other: &Self) {
        self.zero_extend(other.digits.len());
        let (lhs, rhs) = (&mut self.digits, other.digits.as_slice());

        let mut carry = false;
        for (lhs_digit, &rhs_digit) in lhs.iter_mut().zip(rhs) {
            (*lhs_digit, carry) = lhs_digit.carrying_add_ext(rhs_digit, carry);
        }
        for digit in lhs.get_mut(rhs.len()..).unwrap_or_default() {
            if !carry {
                break;
            }
            (*digit, carry) = digit.overflowing_add(1);
        }
        if carry {
            lhs.push(1);
        }
    }
    pub fn borrowing_sub_inplace(&mut self, other: &Self) -> bool {
        self.zero_extend(other.digits.len());
        let (lhs, rhs) = (&mut self.digits, other.digits.as_slice());

        let mut borrow = false;
        for (lhs_digit, &rhs_digit) in lhs.iter_mut().zip(rhs) {
            (*lhs_digit, borrow) = lhs_digit.borrowing_sub_ext(rhs_digit, borrow);
        }
        for digit in lhs.get_mut(rhs.len()..).unwrap_or_default() {
            if !borrow {
                break;
            }
            (*digit, borrow) = digit.overflowing_sub(1);
        }
        self.trim_zeros();
        borrow
    }
    pub fn add(&self, other: &Self) -> Self {
        let mut sum = self.clone_with_capacity(self.num_digits().max(other.num_digits()) + 1);
        sum.add_inplace(other);
        sum
    }
    pub fn checked_sub(&self, other: &Self) -> Option<Self> {
        let mut diff = self.clone();
        let borrow = diff.borrowing_sub_inplace(other);
        (!borrow).then_some(diff)
    }
    pub fn sub_inplace(&mut self, other: &Self) {
        assert!(
            !self.borrowing_sub_inplace(other),
            "Subtraction must not underflow"
        );
    }
    pub fn sub(&self, other: &Self) -> Self {
        self.checked_sub(other)
            .expect("Subtraction must not underflow")
    }
    pub fn inc_inplace(&mut self) {
        for digit in &mut self.digits {
            let carry;
            (*digit, carry) = digit.overflowing_add(1);
            if !carry {
                return;
            }
        }
        self.digits.push(1);
    }
    pub fn dec_inplace(&mut self) -> bool {
        if self.is_zero() {
            return true;
        }
        for digit in &mut self.digits {
            let borrow;
            (*digit, borrow) = digit.overflowing_sub(1);
            if !borrow {
                self.trim_zeros();
                return false;
            }
        }
        unreachable!()
    }
    pub fn and_inplace(&mut self, other: &Self) {
        for (my_digit, &other_digit) in self.digits.iter_mut().zip(&other.digits) {
            *my_digit &= other_digit;
        }
    }
    pub fn or_inplace(&mut self, other: &Self) {
        for (my_digit, &other_digit) in self.digits.iter_mut().zip(&other.digits) {
            *my_digit |= other_digit;
        }
        self.digits
            .extend_from_slice(other.digits.get(self.num_digits()..).unwrap_or_default());
    }
    pub fn xor_inplace(&mut self, other: &Self) {
        for (my_digit, &other_digit) in self.digits.iter_mut().zip(&other.digits) {
            *my_digit ^= other_digit;
        }
        self.digits
            .extend_from_slice(other.digits.get(self.num_digits()..).unwrap_or_default());
    }
    pub fn shl_inplace(&mut self, shift: u64) {
        if self.is_zero() {
            return;
        }

        let o_num_digits = self.num_digits();
        let offset = (shift / 64) as usize;
        if shift % 64 == 0 {
            self.zero_extend(o_num_digits + offset);
            self.digits.copy_within(..o_num_digits, offset);
            self.digits[..offset].fill(0);
            debug_assert_ne!(self.digits.last(), Some(&0));
            return;
        }

        self.zero_extend(o_num_digits + offset + 1);
        for i in (0..o_num_digits).rev() {
            self.digits[i + offset + 1] |= self.digits[i] >> 64 - shift % 64;
            self.digits[i + offset] = self.digits[i] << shift % 64;
        }
        self.digits[..offset].fill(0);
        self.trim_zeros();
    }
    pub fn shl(&self, shift: u64) -> Self {
        if self.is_zero() {
            return Self::ZERO;
        }

        let o_num_digits = self.num_digits();
        let offset = (shift / 64) as usize;
        if shift % 64 == 0 {
            let mut out_digits = vec![0; o_num_digits + offset];
            out_digits[offset..].copy_from_slice(&self.digits);
            debug_assert_ne!(out_digits.last(), Some(&0));
            return Self { digits: out_digits };
        }

        let mut out_digits = vec![0; o_num_digits + offset + 1];
        for i in 0..o_num_digits {
            out_digits[i + offset] |= self.digits[i] << shift % 64;
            out_digits[i + offset + 1] = self.digits[i] >> 64 - shift % 64;
        }
        let mut out = Self { digits: out_digits };
        out.trim_zeros();
        out
    }
    pub fn shr_inplace(&mut self, shift: u64) {
        let o_num_digits = self.num_digits();
        let offset = (shift / 64) as usize;

        if o_num_digits <= offset {
            self.digits.clear();
            return;
        }

        if shift % 64 == 0 {
            self.digits.drain(..offset);
            debug_assert_ne!(self.digits.last(), Some(&0));
            return;
        }

        self.digits[0] = self.digits[offset] >> shift % 64;
        for i in offset + 1..o_num_digits {
            self.digits[i - offset - 1] |= self.digits[i] << 64 - shift % 64;
            self.digits[i - offset] = self.digits[i] >> shift % 64;
        }
        self.digits.truncate(o_num_digits - offset);
        self.trim_zeros();
    }
    pub fn shr(&self, shift: u64) -> Self {
        let o_num_digits = self.num_digits();
        let offset = (shift / 64) as usize;

        if o_num_digits <= offset {
            return Self::ZERO;
        }

        if shift % 64 == 0 {
            let out_digits = self.digits[offset..].to_vec();
            debug_assert_ne!(out_digits.last(), Some(&0));
            return Self { digits: out_digits };
        }

        let mut out_digits = vec![0; o_num_digits - offset];
        out_digits[0] = self.digits[offset] >> shift % 64;
        for i in offset + 1..o_num_digits {
            out_digits[i - offset - 1] |= self.digits[i] << 64 - shift % 64;
            out_digits[i - offset] = self.digits[i] >> shift % 64;
        }
        let mut out = Self { digits: out_digits };
        out.trim_zeros();
        out
    }
    pub fn bit_mask_inplace<R: ops::RangeBounds<u64>>(&mut self, range: R) {
        let start = match range.start_bound() {
            ops::Bound::Included(&start) => start,
            ops::Bound::Excluded(&start) => start.saturating_sub(1),
            ops::Bound::Unbounded => 0,
        };
        let end = self.bit_len().min(match range.end_bound() {
            ops::Bound::Included(&end) => end.saturating_add(1),
            ops::Bound::Excluded(&end) => end,
            ops::Bound::Unbounded => u64::MAX,
        });
        if end <= start {
            self.digits.clear();
            return;
        }
        if end % 64 == 0 {
            self.digits.truncate((end / 64) as _);
        } else {
            self.digits.truncate((end / 64 + 1) as _);
            *self.digits.last_mut().unwrap() &= (1 << end % 64) - 1;
        }
        self.digits[..(start / 64) as usize].fill(0);
        if start % 64 != 0 {
            self.digits[(start / 64) as usize] &= (1u64 << start % 64).wrapping_neg();
        }
        self.trim_zeros();
    }
    pub fn bit_mask<R: ops::RangeBounds<u64>>(&self, range: R) -> Self {
        let start = match range.start_bound() {
            ops::Bound::Included(&start) => start,
            ops::Bound::Excluded(&start) => start.saturating_sub(1),
            ops::Bound::Unbounded => 0,
        };
        let end = self.bit_len().min(match range.end_bound() {
            ops::Bound::Included(&end) => end.saturating_add(1),
            ops::Bound::Excluded(&end) => end,
            ops::Bound::Unbounded => u64::MAX,
        });
        if end <= start {
            return Self::ZERO;
        }
        let num_out_digits = (end / 64) as usize + (end % 64 != 0) as usize;
        let mut out_digits = vec![0; num_out_digits];
        out_digits[(start / 64) as usize..]
            .copy_from_slice(&self.digits[(start / 64) as usize..num_out_digits]);
        if start % 64 != 0 {
            out_digits[(start / 64) as usize] &= (1u64 << start % 64).wrapping_neg();
        }
        if end % 64 != 0 {
            out_digits[(end / 64) as usize] &= (1 << end % 64) - 1;
        }
        let mut out = Self { digits: out_digits };
        out.trim_zeros();
        out
    }
    // /// Slightly slower than [`Self::long_mul`] so prefer that
    // /// unless there are weird memory constraints.
    // pub fn long_mul_inplace(&mut self, other: &Self) {
    //     let my_num_digits = self.num_digits();
    //     self.zero_extend(my_num_digits + other.num_digits());
    //     for i in (0..my_num_digits).rev() {
    //         let my_digit = mem::take(&mut self.digits[i]);

    //         let mut carry_bit = false;
    //         let mut carry = 0u64;
    //         for (product_digit, &other_digit) in self.digits[i..].iter_mut().zip(&other.digits) {
    //             (*product_digit, carry_bit) = product_digit.carrying_add_ext(carry, carry_bit);
    //             (*product_digit, carry) = my_digit.carrying_mul_ext(other_digit, *product_digit);
    //         }
    //         let product_digit = &mut self.digits[i + other.num_digits()];
    //         (*product_digit, carry_bit) = product_digit.carrying_add_ext(carry, carry_bit);
    //         for product_digit in &mut self.digits[i + other.num_digits() + 1..] {
    //             if !carry_bit {
    //                 break;
    //             }
    //             (*product_digit, carry_bit) = product_digit.overflowing_add(1);
    //         }
    //         assert!(!carry_bit)
    //     }
    // }
    pub fn long_mul(&self, other: &Self) -> Self {
        let mut product_digits = vec![0; self.num_digits() + other.num_digits()];
        for (i, &my_digit) in self.digits.iter().enumerate() {
            let mut carry_bit = false;
            let mut carry = 0u64;
            for (product_digit, &other_digit) in product_digits[i..].iter_mut().zip(&other.digits) {
                (*product_digit, carry_bit) = product_digit.carrying_add_ext(carry, carry_bit);
                (*product_digit, carry) = my_digit.carrying_mul_ext(other_digit, *product_digit);
            }
            let product_digit = &mut product_digits[i + other.num_digits()];
            (*product_digit, carry_bit) = product_digit.carrying_add_ext(carry, carry_bit);
            assert!(!carry_bit);
        }
        Self {
            digits: product_digits,
        }
    }
    pub fn karatsuba_mul(&self, other: &Self) -> Self {
        let num_product_digits = (self.num_digits() + other.num_digits()) as u64;
        let midpoint =
            (num_product_digits / 4 + (num_product_digits % 4 != 0) as u64).saturating_mul(64);

        let self0 = self.bit_mask(..midpoint);
        let self1 = self.shr(midpoint);

        let other0 = other.bit_mask(..midpoint);
        let other1 = other.shr(midpoint);

        let prod0 = self0.mul(&other0);
        let prod2 = self1.mul(&other1);

        let mut self01 = self0;
        self01.add_inplace(&self1);

        let mut other01 = other0;
        other01.add_inplace(&other1);

        let mut prod1 = self01.mul(&other01);
        prod1.sub_inplace(&prod2);
        prod1.sub_inplace(&prod0);

        let mut prod = prod2;
        prod.shl_inplace(midpoint);
        prod.add_inplace(&prod1);

        prod.shl_inplace(midpoint);
        prod.add_inplace(&prod0);

        prod
    }
    pub fn schönhage_strassen_mul(&self, other: &Self) -> Self {
        /// Calculates `self % (2**mod_digits + 1)` and stores it into self.
        /// Can be applied only to values less than or equal to `(2**mod_digits)**2`.
        ///
        /// # Panics
        /// If `self` is larger than `(2**mod_digits)**2`.
        fn mod_fermat_inplace(slf: &mut BigUint, mod_digits: usize) {
            if slf.num_digits() <= mod_digits {
                return;
            } else if slf.num_digits() <= 2 * mod_digits {
                let (head_digits, tail_digits) = slf.digits.split_at_mut(mod_digits);

                let mut borrow = false;
                for (head_digit, &tail_digit) in head_digits.iter_mut().zip(&*tail_digits) {
                    (*head_digit, borrow) = head_digit.borrowing_sub_ext(tail_digit, borrow);
                }
                for head_digit in &mut head_digits[tail_digits.len()..] {
                    if !borrow {
                        break;
                    }
                    (*head_digit, borrow) = head_digit.overflowing_sub(1);
                }
                slf.digits.truncate(mod_digits);
                slf.trim_zeros();
                if borrow {
                    slf.inc_inplace();
                }
            } else {
                let (&tail_digit, head_digits) = slf.digits.split_last().unwrap();
                assert!(
                    tail_digit == 1
                        && head_digits.len() == 2 * mod_digits
                        && head_digits.iter().all(|&digit| digit == 0),
                    "`self` must be less than or equal to `(2**mod_digits)**2`"
                );
                slf.digits.clear();
                slf.digits.push(1);
                return;
            }
        }
        /// Subtracts `other` from `self` and applys `mod_fermat_inplace` before and after.
        ///
        /// # Requirements
        /// All of `mod_fermat_inplace`'s requirements must be held.
        /// In addition, `other` must be less than or equal to `2 ** mod_digits`,
        /// otherwise the subtraction may underflow and panic.
        fn sub_mod_fermat_inplace(slf: &mut BigUint, other: &BigUint, mod_digits: usize) {
            mod_fermat_inplace(slf, mod_digits);
            if slf.num_digits() <= mod_digits {
                slf.zero_extend(mod_digits + 1);
                slf.digits[mod_digits] = 1;
                slf.inc_inplace();
            }
            slf.sub_inplace(other);
            mod_fermat_inplace(slf, mod_digits);
        }
        fn fft(blocks: &mut [BigUint], block_digits: usize) -> usize {
            assert!(
                (blocks.iter()).all(|block| block.num_digits() <= block_digits),
                "All blocks must have at most `block_digits` digits",
            );
            assert!(1 < blocks.len());
            assert!(blocks.len().is_power_of_two());

            let log_num_blocks = blocks.len().ilog2() as usize;
            let m_tag = (block_digits >> log_num_blocks - 1) + 1;
            let mod_digits = m_tag << log_num_blocks;

            for step in (0..log_num_blocks).rev() {
                let halfway = 1 << step;
                for i in (0..blocks.len()).step_by(2 * halfway) {
                    for j in 0..halfway {
                        // t := (a[i+j] - a[i+j+halfway]) * root^(-j * 2^(log_len_a - step))
                        // a[i + j] += a[i+j+halfway]
                        // a[i+j+halfway] = t

                        let mut t = blocks[i + j + halfway].clone_with_capacity(2 * mod_digits + 1);
                        sub_mod_fermat_inplace(&mut t, &blocks[i + j], mod_digits);
                        t.shl_inplace(
                            64 * (mod_digits - (j * m_tag << log_num_blocks - step)) as u64,
                        );
                        mod_fermat_inplace(&mut t, mod_digits);

                        let t = mem::replace(&mut blocks[i + j + halfway], t);
                        blocks[i + j].add_inplace(&t);
                        mod_fermat_inplace(&mut blocks[i + j], mod_digits);
                    }
                }
            }

            mod_digits
        }
        fn ifft(blocks: &mut [BigUint], block_digits: usize) {
            assert!(1 < blocks.len());
            assert!(blocks.len().is_power_of_two());

            let num_blocks = blocks.len();
            let log_num_blocks = num_blocks.ilog2() as usize;
            let m_tag = (block_digits >> log_num_blocks - 1) + 1;
            let mod_digits = m_tag << log_num_blocks;

            assert!(
                (blocks.iter()).all(|block| block.num_digits() <= mod_digits + 1),
                "All blocks must have at most `mod_digits + 1` digits",
            );

            for step in 0..log_num_blocks {
                let halfway = 1 << step;
                for i in (0..blocks.len()).step_by(2 * halfway) {
                    for j in 0..halfway {
                        let mut t = blocks[i + j].clone_with_capacity(mod_digits + 2);
                        t = mem::replace(&mut blocks[i + j + halfway], t);
                        t.shl_inplace(64 * (j * m_tag << log_num_blocks - step) as u64);
                        mod_fermat_inplace(&mut t, mod_digits);

                        sub_mod_fermat_inplace(&mut blocks[i + j + halfway], &t, mod_digits);
                        blocks[i + j].add_inplace(&t);
                        mod_fermat_inplace(&mut blocks[i + j], mod_digits);
                    }
                }
            }

            for block in blocks {
                if block.is_zero() {
                    continue;
                }
                assert_eq!(block.digits()[0] & num_blocks as u64 - 1, 0);
                block.shr_inplace(log_num_blocks as _);
            }
        }

        if self.is_zero() || other.is_zero() {
            return Self::ZERO;
        }

        let num_prod_digits = self.num_digits() + other.num_digits();
        let log_num_blocks = (num_prod_digits / 0x400).ilog2() as usize;
        let num_blocks = 1usize << log_num_blocks;
        let block_digits = (num_prod_digits - 1 >> log_num_blocks) + 1;
        assert!(num_prod_digits <= block_digits << log_num_blocks);

        println!("SCHÖNHAGE_STRASSEN[PARAMS]:  k={log_num_blocks}  M={block_digits}*2⁶⁴");

        use std::time::Instant;

        let fft1_start = Instant::now();

        let mut my_blocks = vec![Self::ZERO; num_blocks];
        for (digits, block) in self.digits.chunks(block_digits).zip(&mut my_blocks) {
            block.digits.extend_from_slice(digits);
            block.trim_zeros();
        }
        let mod_digits = fft(&mut my_blocks, block_digits);

        println!("                             n'={mod_digits}*2⁶⁴");
        println!("  FFT1: {:?}", fft1_start.elapsed());
        let fft2_start = Instant::now();

        let mut other_blocks = vec![Self::ZERO; num_blocks];
        for (digits, block) in other.digits.chunks(block_digits).zip(&mut other_blocks) {
            block.digits.extend_from_slice(digits);
            block.trim_zeros();
        }
        assert_eq!(fft(&mut other_blocks, block_digits), mod_digits);

        println!("  FFT2: {:?}", fft2_start.elapsed());
        let prod_start = Instant::now();

        let mut prod_blocks: Vec<_> = (my_blocks.iter().zip(&other_blocks))
            .map(|(my_block, other_block)| {
                let mut block = my_block.mul(other_block);
                mod_fermat_inplace(&mut block, mod_digits);
                block
            })
            .collect();

        println!("  PROD: {:?}", prod_start.elapsed());
        let ifft_start = Instant::now();

        ifft(&mut prod_blocks, block_digits);

        println!("  IFFT: {:?}", ifft_start.elapsed());
        let carry_start = Instant::now();

        assert!(prod_blocks[(num_prod_digits - 1) / block_digits + 1..]
            .iter()
            .all(BigUint::is_zero));

        let mut prod_digits = vec![0; num_prod_digits];
        let mut carry = BigUint::ZERO;
        for (digits, block) in prod_digits.chunks_mut(block_digits).zip(prod_blocks) {
            carry.add_inplace(&block);
            let cpy_len = carry.num_digits().min(block_digits);
            digits[..cpy_len].copy_from_slice(&carry.digits[..cpy_len]);
            carry.shr_inplace(64 * block_digits as u64);
        }
        assert!(carry.is_zero());

        println!("  CARRY: {:?}", carry_start.elapsed());

        let mut prod = Self {
            digits: prod_digits,
        };
        prod.trim_zeros();
        prod

        // return int.from_bytes(c_ds, "little")
    }
    pub fn mul(&self, other: &Self) -> Self {
        match self.num_digits() + other.num_digits() {
            ..0x40 => self.long_mul(other),
            ..0x20000 => self.karatsuba_mul(other),
            _ => self.schönhage_strassen_mul(other),
        }
    }
}

impl PartialOrd for BigUint {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for BigUint {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.digits.len().cmp(&other.digits.len()))
            .then_with(|| self.digits.iter().rev().cmp(other.digits.iter().rev()))
    }
}

#[derive(Clone, PartialEq, Eq, Default)]
pub struct BigInt {
    value: BigUint,
    is_neg: bool,
}

impl BigInt {
    pub const ZERO: Self = Self {
        value: BigUint::ZERO,
        is_neg: false,
    };
    pub const fn zero() -> Self {
        Self::ZERO
    }
    pub fn from_i64(x: i64) -> Self {
        Self {
            value: BigUint::from_u64(x.unsigned_abs()),
            is_neg: x.is_negative(),
        }
    }
    pub fn from_u64(x: u64) -> Self {
        Self {
            value: BigUint::from_u64(x),
            is_neg: false,
        }
    }
    pub fn with_capacity(num_digits: usize) -> Self {
        Self {
            value: BigUint::with_capacity(num_digits),
            is_neg: false,
        }
    }
    pub fn clone_with_capacity(&self, num_digits: usize) -> Self {
        Self {
            value: self.value.clone_with_capacity(num_digits),
            is_neg: self.is_neg,
        }
    }
    pub fn is_zero(&self) -> bool {
        self.value.is_zero()
    }
    pub fn is_negative(&self) -> bool {
        self.is_neg
    }
    pub fn is_positive(&self) -> bool {
        !self.is_negative() && !self.is_zero()
    }
    pub fn num_digits(&self) -> usize {
        self.value.num_digits()
    }
    pub fn bit_len(&self) -> u64 {
        self.value.bit_len()
    }
    // O(n)
    fn trim_zeros(&mut self) {
        self.value.trim_zeros();
        self.is_neg &= !self.is_zero();
    }
    pub fn neg_inplace(&mut self) {
        self.trim_zeros();
        self.is_neg ^= !self.is_zero();
    }
    pub fn neg(mut self) -> Self {
        self.neg_inplace();
        self
    }
    pub fn add_inplace(&mut self, other: &Self) {
        if self.is_neg == other.is_neg {
            self.value.add_inplace(&other.value);
            return;
        }
        if self.value.borrowing_sub_inplace(&other.value) {
            let mut carry = true;
            for digit in &mut self.value.digits {
                (*digit, carry) = (!*digit).overflowing_add(carry as _);
            }
            if carry {
                self.value.digits.push(1);
            }
            self.neg_inplace();
        }
    }
    pub fn sub_inplace(&mut self, other: &Self) {
        self.neg_inplace();
        self.add_inplace(other);
        self.neg_inplace();
    }
    pub fn add(&self, other: &Self) -> Self {
        let mut sum = self.clone_with_capacity(self.num_digits().max(other.num_digits()) + 1);
        sum.add_inplace(other);
        sum
    }
    pub fn sub(&self, other: &Self) -> Self {
        let mut diff = self.clone_with_capacity(self.num_digits().max(other.num_digits()) + 1);
        diff.sub_inplace(other);
        diff
    }
    pub fn inc_inplace(&mut self) {
        if self.is_neg {
            self.value.dec_inplace();
            self.is_neg &= !self.is_zero();
        } else {
            self.value.inc_inplace();
        }
    }
    pub fn dec_inplace(&mut self) {
        if self.is_neg {
            self.value.inc_inplace();
        } else {
            self.value.dec_inplace();
        }
    }
    pub fn shl_inplace(&mut self, shift: u64) {
        self.value.shl_inplace(shift);
    }
    pub fn shr_inplace(&mut self, shift: u64) {
        if self.is_neg {
            self.value.dec_inplace();
            self.value.shr_inplace(shift);
            self.value.inc_inplace();
        } else {
            self.value.shr_inplace(shift);
        }
    }
    pub fn not_inplace(&mut self) {
        self.inc_inplace();
        self.neg_inplace();
    }
    pub fn and_inplace(&mut self, other: &Self) {
        match (self.is_neg, other.is_neg) {
            (false, false) => self.value.and_inplace(&other.value),
            (false, true) => {
                let my_digits = &mut self.value.digits;
                let other_digits = other.value.digits.as_slice();

                let mut other_borrow = true;
                for (my_digit, &(mut other_digit)) in my_digits.iter_mut().zip(other_digits) {
                    (other_digit, other_borrow) = other_digit.overflowing_sub(other_borrow as _);
                    *my_digit &= !other_digit;
                }
            }
            (true, false) => {
                self.value.dec_inplace();
                self.value.zero_extend(other.num_digits());
                let my_digits = &mut self.value.digits;
                let other_digits = other.value.digits.as_slice();
                for (my_digit, &other_digit) in my_digits.iter_mut().zip(other_digits) {
                    *my_digit = !*my_digit & other_digit;
                }
                self.is_neg = false;
            }
            (true, true) => {
                self.value.dec_inplace();
                self.value.zero_extend(other.num_digits());
                let my_digits = &mut self.value.digits;
                let other_digits = other.value.digits.as_slice();

                let mut other_borrow = true;
                for (my_digit, &(mut other_digit)) in my_digits.iter_mut().zip(other_digits) {
                    (other_digit, other_borrow) = other_digit.overflowing_sub(other_borrow as _);
                    *my_digit |= other_digit;
                }
                self.value.inc_inplace();
            }
        }
        self.trim_zeros();
    }
    pub fn and(&mut self, other: &Self) -> Self {
        let mut res = self.clone_with_capacity(self.num_digits().max(other.num_digits()) + 1);
        res.and_inplace(other);
        res
    }
    pub fn or_inplace(&mut self, other: &Self) {
        match (self.is_neg, other.is_neg) {
            (false, false) => self.value.or_inplace(&other.value),
            (false, true) => {
                let my_digits = &mut self.value.digits;
                let other_digits = other.value.digits.as_slice();
                my_digits.truncate(other_digits.len());

                let mut other_borrow = true;
                for (my_digit, &(mut other_digit)) in my_digits.iter_mut().zip(other_digits) {
                    (other_digit, other_borrow) = other_digit.overflowing_sub(other_borrow as _);
                    *my_digit = !*my_digit & other_digit;
                }
                self.value.inc_inplace();
                self.is_neg = true;
            }
            (true, false) => {
                self.value.dec_inplace();
                let my_digits = &mut self.value.digits;
                let other_digits = other.value.digits.as_slice();
                for (my_digit, &other_digit) in my_digits.iter_mut().zip(other_digits) {
                    *my_digit = *my_digit & !other_digit;
                }
                self.value.inc_inplace();
            }
            (true, true) => {
                self.value.dec_inplace();
                let my_digits = &mut self.value.digits;
                let other_digits = other.value.digits.as_slice();
                my_digits.truncate(other_digits.len());

                let mut other_borrow = true;
                for (my_digit, &(mut other_digit)) in my_digits.iter_mut().zip(other_digits) {
                    (other_digit, other_borrow) = other_digit.overflowing_sub(other_borrow as _);
                    *my_digit &= other_digit;
                }
                self.value.inc_inplace();
            }
        }
        self.trim_zeros();
    }
    pub fn or(&mut self, other: &Self) -> Self {
        let mut res = self.clone_with_capacity(self.num_digits().max(other.num_digits()) + 1);
        res.or_inplace(other);
        res
    }
    pub fn xor_inplace(&mut self, other: &Self) {
        match (self.is_neg, other.is_neg) {
            (false, false) => self.value.xor_inplace(&other.value),
            (false, true) => {
                self.value.zero_extend(other.num_digits());

                let my_digits = &mut self.value.digits;
                let other_digits = other.value.digits.as_slice();

                let mut other_borrow = true;
                for (my_digit, &(mut other_digit)) in my_digits.iter_mut().zip(other_digits) {
                    (other_digit, other_borrow) = other_digit.overflowing_sub(other_borrow as _);
                    *my_digit ^= other_digit;
                }
                self.value.inc_inplace();
                self.is_neg = true;
            }
            (true, false) => {
                self.value.dec_inplace();
                self.value.zero_extend(other.num_digits());

                let my_digits = &mut self.value.digits;
                let other_digits = other.value.digits.as_slice();

                for (my_digit, &other_digit) in my_digits.iter_mut().zip(other_digits) {
                    *my_digit ^= other_digit;
                }
                self.value.inc_inplace();
            }
            (true, true) => {
                self.value.dec_inplace();
                self.value.zero_extend(other.num_digits());

                let my_digits = &mut self.value.digits;
                let other_digits = other.value.digits.as_slice();

                let mut other_borrow = true;
                for (my_digit, &(mut other_digit)) in my_digits.iter_mut().zip(other_digits) {
                    (other_digit, other_borrow) = other_digit.overflowing_sub(other_borrow as _);
                    *my_digit ^= other_digit;
                }
                self.is_neg = false;
            }
        }
        self.trim_zeros();
    }
    pub fn xor(&mut self, other: &Self) -> Self {
        let mut res = self.clone_with_capacity(self.num_digits().max(other.num_digits()) + 1);
        res.xor_inplace(other);
        res
    }
    pub fn bit_mask_inplace<R: ops::RangeBounds<u64>>(&mut self, range: R) {
        if self.is_neg {
            self.is_neg = false;
            self.value.dec_inplace();
            for digit in &mut self.value.digits {
                *digit = !*digit;
            }
        }
        self.value.bit_mask_inplace(range);
    }
    pub fn mul(&self, other: &Self) -> Self {
        Self {
            value: self.value.mul(&other.value),
            is_neg: self.is_neg ^ other.is_neg,
        }
    }
}

impl PartialOrd for BigInt {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BigInt {
    fn cmp(&self, other: &Self) -> Ordering {
        self.is_neg.cmp(&other.is_neg).reverse().then_with(|| {
            let ord = self.value.cmp(&other.value);
            match self.is_neg {
                false => ord,
                true => ord.reverse(),
            }
        })
    }
}

impl fmt::LowerHex for BigUint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0x{:x}", self.digits.last().unwrap_or(&0))?;
        for &digit in self.digits.iter().rev().skip(1) {
            write!(f, "_{digit:0>16x}")?;
        }

        Ok(())
    }
}
impl fmt::LowerHex for BigInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use fmt::Write;

        if self.is_negative() {
            f.write_char('-')?;
        } else if f.sign_plus() {
            f.write_char('+')?;
        }

        fmt::LowerHex::fmt(&self.value, f)
    }
}
