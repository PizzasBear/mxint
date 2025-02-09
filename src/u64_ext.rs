#[allow(dead_code)]
pub(crate) trait U64Ext: Sized {
    fn carrying_add_ext(self, rhs: Self, carry: bool) -> (Self, bool);
    fn borrowing_sub_ext(self, rhs: Self, borrow: bool) -> (Self, bool);
    fn widening_mul_ext(self, rhs: Self) -> (Self, Self);
    fn carrying_mul_ext(self, rhs: Self, carry: Self) -> (Self, Self);
    fn midpoint_ext(self, rhs: Self) -> Self;
}

impl U64Ext for u64 {
    /// Calculates `self` + `rhs` + `carry` and returns a tuple containing
    /// the sum and the output carry.
    ///
    /// Performs "ternary addition" of two integer operands and a carry-in
    /// bit, and returns an output integer and a carry-out bit. This allows
    /// chaining together multiple additions to create a wider addition, and
    /// can be useful for bignum addition.
    ///
    /// This can be thought of as a 64-bit "full adder", in the electronics sense.
    ///
    /// If the input carry is false, this method is equivalent to
    /// [`overflowing_add`](Self::overflowing_add), and the output carry is
    /// equal to the overflow flag. Note that although carry and overflow
    /// flags are similar for unsigned integers, they are different for
    /// signed integers.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(bigint_helper_methods)]
    ///
    /// //    3  MAX    (a = 3 × 2^64 + 2^64 - 1)
    /// // +  5    7    (b = 5 × 2^64 + 7)
    /// // ---------
    /// //    9    6    (sum = 9 × 2^64 + 6)
    ///
    /// let (a1, a0): (u64, u64) = (3, u64::MAX);
    /// let (b1, b0): (u64, u64) = (5, 7);
    /// let carry0 = false;
    ///
    /// let (sum0, carry1) = a0.carrying_add_ext(b0, carry0);
    /// assert_eq!(carry1, true);
    /// let (sum1, carry2) = a1.carrying_add_ext(b1, carry1);
    /// assert_eq!(carry2, false);
    ///
    /// assert_eq!((sum1, sum0), (9, 6));
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    fn carrying_add_ext(self, rhs: Self, carry: bool) -> (Self, bool) {
        // note: longer-term this should be done via an intrinsic, but this has been shown
        //   to generate optimal code for now, and LLVM doesn't have an equivalent intrinsic
        let (a, b) = self.overflowing_add(rhs);
        let (c, d) = a.overflowing_add(carry as _);
        (c, b || d)
    }

    /// Calculates `self` &minus; `rhs` &minus; `borrow` and returns a tuple
    /// containing the difference and the output borrow.
    ///
    /// Performs "ternary subtraction" by subtracting both an integer
    /// operand and a borrow-in bit from `self`, and returns an output
    /// integer and a borrow-out bit. This allows chaining together multiple
    /// subtractions to create a wider subtraction, and can be useful for
    /// bignum subtraction.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(bigint_helper_methods)]
    ///
    /// //    9    6    (a = 9 × 2^64 + 6)
    /// // -  5    7    (b = 5 × 2^64 + 7)
    /// // ---------
    /// //    3  MAX    (diff = 3 × 2^64 + 2^64 - 1)
    ///
    /// let (a1, a0): (u64, u64) = (9, 6);
    /// let (b1, b0): (u64, u64) = (5, 7);
    /// let borrow0 = false;
    ///
    /// let (diff0, borrow1) = a0.borrowing_sub_ext(b0, borrow0);
    /// assert_eq!(borrow1, true);
    /// let (diff1, borrow2) = a1.borrowing_sub_ext(b1, borrow1);
    /// assert_eq!(borrow2, false);
    ///
    /// assert_eq!((diff1, diff0), (3, u64::MAX));
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    fn borrowing_sub_ext(self, rhs: Self, borrow: bool) -> (Self, bool) {
        // note: longer-term this should be done via an intrinsic, but this has been shown
        //   to generate optimal code for now, and LLVM doesn't have an equivalent intrinsic
        let (a, b) = self.overflowing_sub(rhs);
        let (c, d) = a.overflowing_sub(borrow as _);
        (c, b || d)
    }

    /// Calculates the complete product `self * rhs` without the possibility to overflow.
    ///
    /// This returns the low-order (wrapping) bits and the high-order (overflow) bits
    /// of the result as two separate values, in that order.
    ///
    /// If you also need to add a carry to the wide result, then you want
    /// [`Self::carrying_mul_ext`] instead.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// Please note that this example is shared between integer types.
    /// Which explains why `u32` is used here.
    ///
    /// ```
    /// #![feature(bigint_helper_methods)]
    /// assert_eq!(5u32.widening_mul_ext(2), (10, 0));
    /// assert_eq!(1_000_000_000u32.widening_mul_ext(10), (1410065408, 2));
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    fn widening_mul_ext(self, rhs: Self) -> (Self, Self) {
        // note: longer-term this should be done via an intrinsic,
        //   but for now we can deal without an impl for u128/i128
        // SAFETY: overflow will be contained within the wider types
        let wide = unsafe { (self as u128).unchecked_mul(rhs as u128) };
        (wide as _, (wide >> Self::BITS) as _)
    }

    /// Calculates the "full multiplication" `self * rhs + carry`
    /// without the possibility to overflow.
    ///
    /// This returns the low-order (wrapping) bits and the high-order (overflow) bits
    /// of the result as two separate values, in that order.
    ///
    /// Performs "long multiplication" which takes in an extra amount to add, and may return an
    /// additional amount of overflow. This allows for chaining together multiple
    /// multiplications to create "big integers" which represent larger values.
    ///
    /// If you don't need the `carry`, then you can use [`Self::widening_mul_ext`] instead.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// Please note that this example is shared between integer types.
    /// Which explains why `u32` is used here.
    ///
    /// ```
    /// #![feature(bigint_helper_methods)]
    /// assert_eq!(5u32.carrying_mul_ext(2, 0), (10, 0));
    /// assert_eq!(5u32.carrying_mul_ext(2, 10), (20, 0));
    /// assert_eq!(1_000_000_000u32.carrying_mul_ext(10, 0), (1410065408, 2));
    /// assert_eq!(1_000_000_000u32.carrying_mul_ext(10, 10), (1410065418, 2));
    /// assert_eq!(u64::MAX.carrying_mul_ext(u64::MAX, u64::MAX), (0, u64::MAX));
    /// ```
    ///
    /// This is the core operation needed for scalar multiplication when
    /// implementing it for wider-than-native types.
    ///
    /// ```
    /// #![feature(bigint_helper_methods)]
    /// fn scalar_mul_eq(little_endian_digits: &mut Vec<u16>, multiplicand: u16) {
    ///     let mut carry = 0;
    ///     for d in little_endian_digits.iter_mut() {
    ///         (*d, carry) = d.carrying_mul_ext(multiplicand, carry);
    ///     }
    ///     if carry != 0 {
    ///         little_endian_digits.push(carry);
    ///     }
    /// }
    ///
    /// let mut v = vec![10, 20];
    /// scalar_mul_eq(&mut v, 3);
    /// assert_eq!(v, [30, 60]);
    ///
    /// assert_eq!(0x87654321_u64 * 0xFEED, 0x86D3D159E38D);
    /// let mut v = vec![0x4321, 0x8765];
    /// scalar_mul_eq(&mut v, 0xFEED);
    /// assert_eq!(v, [0xE38D, 0xD159, 0x86D3]);
    /// ```
    ///
    /// If `carry` is zero, this is similar to [`overflowing_mul`](Self::overflowing_mul),
    /// except that it gives the value of the overflow instead of just whether one happened:
    ///
    /// ```
    /// #![feature(bigint_helper_methods)]
    /// let r = u8::carrying_mul_ext(7, 13, 0);
    /// assert_eq!((r.0, r.1 != 0), u8::overflowing_mul(7, 13));
    /// let r = u8::carrying_mul_ext(13, 42, 0);
    /// assert_eq!((r.0, r.1 != 0), u8::overflowing_mul(13, 42));
    /// ```
    ///
    /// The value of the first field in the returned tuple matches what you'd get
    /// by combining the [`wrapping_mul`](Self::wrapping_mul) and
    /// [`wrapping_add`](Self::wrapping_add) methods:
    ///
    /// ```
    /// #![feature(bigint_helper_methods)]
    /// assert_eq!(
    ///     789_u16.carrying_mul_ext(456, 123).0,
    ///     789_u16.wrapping_mul(456).wrapping_add(123),
    /// );
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    fn carrying_mul_ext(self, rhs: Self, carry: Self) -> (Self, Self) {
        // note: longer-term this should be done via an intrinsic,
        //   but for now we can deal without an impl for u128/i128
        // SAFETY: overflow will be contained within the wider types
        let wide = unsafe {
            (self as u128)
                .unchecked_mul(rhs as _)
                .unchecked_add(carry as _)
        };
        (wide as _, (wide >> Self::BITS) as _)
    }
    /// Calculates the middle point of `self` and `rhs`.
    ///
    /// `midpoint(a, b)` is `(a + b) >> 1` as if it were performed in a
    /// sufficiently-large signed integral type. This implies that the result is
    /// always rounded towards negative infinity and that no overflow will ever occur.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(num_midpoint)]
    /// assert_eq!(0u64.midpoint(4), 2);
    /// assert_eq!(1u64.midpoint(4), 2);
    /// ```
    #[must_use = "this returns the result of the operation, \
                    without modifying the original"]
    #[inline]
    fn midpoint_ext(self, rhs: Self) -> Self {
        // Use the well known branchless algorithm from Hacker's Delight to compute
        // `(a + b) / 2` without overflowing: `((a ^ b) >> 1) + (a & b)`.
        ((self ^ rhs) >> 1) + (self & rhs)
    }
}
