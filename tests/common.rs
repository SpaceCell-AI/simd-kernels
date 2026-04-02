//! Test helpers with base structures
use minarrow::traits::type_unions::{Float, Integer};
use minarrow::{
    Array, BooleanArray, CategoricalArray, FloatArray, IntegerArray, MaskedArray, NumericArray,
    StringArray, TextArray,
};
#[cfg(feature = "datetime")]
use minarrow::{DatetimeArray, TemporalArray};

/// Dense and Nullable versions of every Array type
pub struct TestColumns {
    pub i32_dense: Array,
    pub i32_nulls: Array,
    pub i64_dense: Array,
    pub i64_nulls: Array,
    pub u32_dense: Array,
    pub u32_nulls: Array,
    pub u64_dense: Array,
    pub u64_nulls: Array,
    pub f32_dense: Array,
    pub f32_nulls: Array,
    pub f64_dense: Array,
    pub f64_nulls: Array,
    pub bool_dense: Array,
    pub bool_nulls: Array,
    pub str_dense: Array,
    pub str_nulls: Array,
    #[cfg(feature = "large_string")]
    pub lstr_dense: Array,
    #[cfg(feature = "large_string")]
    pub lstr_nulls: Array,
    pub dict_dense: Array,
    pub dict_nulls: Array,
    #[cfg(feature = "datetime")]
    pub dt32_dense: Array,
    #[cfg(feature = "datetime")]
    pub dt32_nulls: Array,
    #[cfg(feature = "datetime")]
    pub dt64_dense: Array,
    #[cfg(feature = "datetime")]
    pub dt64_nulls: Array,
}

impl TestColumns {
    /// build once - reuse in every `#[test]`
    pub fn new() -> Self {
        // numeric helpers
        fn int_arr<T: Integer>(vals: &[T]) -> IntegerArray<T> {
            IntegerArray::<T>::from_slice(vals)
        }
        fn int_arr_null<T: Integer>(vals: &[Option<T>]) -> IntegerArray<T> {
            let mut a = IntegerArray::<T>::with_capacity(vals.len(), true);
            for v in vals {
                match v {
                    Some(x) => a.push(*x),
                    None => a.push_null(),
                }
            }
            a
        }
        fn float_arr<T: Float>(vals: &[T]) -> FloatArray<T> {
            FloatArray::<T>::from_slice(vals)
        }
        fn float_arr_null<T: Float>(vals: &[Option<T>]) -> FloatArray<T> {
            let mut a = FloatArray::<T>::with_capacity(vals.len(), true);
            for v in vals {
                match v {
                    Some(x) => a.push(*x),
                    None => a.push_null(),
                }
            }
            a
        }

        // concrete fixtures
        let dense_i32 = int_arr(&[1, 2, 3, 4, 5]);
        let null_i32 = int_arr_null(&[Some(1), None, Some(3), Some(4), None]);

        let dense_i64 = int_arr(&[10_i64, 20, 30, 40, 50]);
        let null_i64 = int_arr_null(&[Some(10_i64), None, Some(30), None, Some(50)]);

        let dense_u32 = int_arr(&[1_u32, 2, 3, 4, 5]);
        let null_u32 = int_arr_null(&[None, Some(2_u32), Some(3), None, Some(5)]);

        let dense_u64 = int_arr(&[100_u64, 200, 300, 400, 500]);
        let null_u64 = int_arr_null(&[Some(100_u64), None, None, Some(400), Some(500)]);

        let dense_f32 = float_arr(&[1.0_f32, 2.0, 3.0, 4.0, 5.0]);
        let null_f32 = float_arr_null(&[Some(1.0_f32), None, Some(3.0), None, Some(5.0)]);

        let dense_f64 = float_arr(&[1.0_f64, 2.0, 3.0, 4.0, 5.0]);
        let null_f64 = float_arr_null(&[None, Some(2.0_f64), Some(3.0), Some(4.0), None]);

        let dense_bool = BooleanArray::from_slice(&[true, false, true, true, false]);
        let mut null_bool = BooleanArray::from_slice(&[true, false, true, false, true]);
        null_bool.set_null(3);

        let dense_str = StringArray::<u32>::from_slice(&["a", "b", "c", "d", "e"]);
        let mut null_str = StringArray::<u32>::from_slice(&["a", "b", "c", "d", "e"]);
        null_str.set_null(1);
        null_str.set_null(4);

        #[cfg(feature = "large_string")]
        let (lstr_dense, lstr_nulls) = {
            let dense = StringArray::<u64>::from_slice(&["lA", "lB", "lC", "lD"]);
            let mut n = dense.clone();
            n.set_null(2);
            (dense, n)
        };

        #[cfg(feature = "default_categorical_8")]
        let dict_dense = {
            let mut d = CategoricalArray::<u8>::default();
            for s in ["x", "y", "x", "z", "y"] {
                d.push_str(s);
            }
            Array::TextArray(TextArray::Categorical8(d.into()))
        };
        #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
        let dict_dense = {
            let mut d = CategoricalArray::<u32>::default();
            for s in ["x", "y", "x", "z", "y"] {
                d.push_str(s);
            }
            Array::TextArray(TextArray::Categorical32(d.into()))
        };
        #[cfg(feature = "default_categorical_8")]
        let dict_nulls = {
            let mut d = {
                let mut arr = CategoricalArray::<u8>::default();
                for s in ["x", "y", "x", "z", "y"] {
                    arr.push_str(s);
                }
                arr
            };
            d.set_null(3);
            Array::TextArray(TextArray::Categorical8(d.into()))
        };
        #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
        let dict_nulls = {
            let mut d = {
                let mut arr = CategoricalArray::<u32>::default();
                for s in ["x", "y", "x", "z", "y"] {
                    arr.push_str(s);
                }
                arr
            };
            d.set_null(3);
            Array::TextArray(TextArray::Categorical32(d.into()))
        };
        #[cfg(feature = "datetime")]
        let (dt32_dense, dt32_nulls, dt64_dense, dt64_nulls) = {
            use minarrow::enums::time_units::TimeUnit::*;
            let d32 = DatetimeArray::<i32>::from_slice(
                &[0, 86_400, 172_800, 259_200, 345_600],
                Some(Days),
            );
            let mut n32 = d32.clone();
            n32.set_null(2);
            let d64 = DatetimeArray::<i64>::from_slice(
                &[1_000, 2_000, 3_000, 4_000, 5_000],
                Some(Milliseconds),
            );
            let mut n64 = d64.clone();
            n64.set_null(0);
            (
                Array::TemporalArray(TemporalArray::Datetime32(d32.into())),
                Array::TemporalArray(TemporalArray::Datetime32(n32.into())),
                Array::TemporalArray(TemporalArray::Datetime64(d64.into())),
                Array::TemporalArray(TemporalArray::Datetime64(n64.into())),
            )
        };

        Self {
            i32_dense: Array::NumericArray(NumericArray::Int32(dense_i32.into())),
            i32_nulls: Array::NumericArray(NumericArray::Int32(null_i32.into())),
            i64_dense: Array::NumericArray(NumericArray::Int64(dense_i64.into())),
            i64_nulls: Array::NumericArray(NumericArray::Int64(null_i64.into())),
            u32_dense: Array::NumericArray(NumericArray::UInt32(dense_u32.into())),
            u32_nulls: Array::NumericArray(NumericArray::UInt32(null_u32.into())),
            u64_dense: Array::NumericArray(NumericArray::UInt64(dense_u64.into())),
            u64_nulls: Array::NumericArray(NumericArray::UInt64(null_u64.into())),
            f32_dense: Array::NumericArray(NumericArray::Float32(dense_f32.into())),
            f32_nulls: Array::NumericArray(NumericArray::Float32(null_f32.into())),
            f64_dense: Array::NumericArray(NumericArray::Float64(dense_f64.into())),
            f64_nulls: Array::NumericArray(NumericArray::Float64(null_f64.into())),
            bool_dense: Array::BooleanArray(dense_bool.into()),
            bool_nulls: Array::BooleanArray(null_bool.into()),
            str_dense: Array::TextArray(TextArray::String32(dense_str.into())),
            str_nulls: Array::TextArray(TextArray::String32(null_str.into())),
            #[cfg(feature = "large_string")]
            lstr_dense: Array::TextArray(TextArray::String64(lstr_dense.into())),
            #[cfg(feature = "large_string")]
            lstr_nulls: Array::TextArray(TextArray::String64(lstr_nulls.into())),
            dict_dense,
            dict_nulls,
            #[cfg(feature = "datetime")]
            dt32_dense,
            #[cfg(feature = "datetime")]
            dt32_nulls,
            #[cfg(feature = "datetime")]
            dt64_dense,
            #[cfg(feature = "datetime")]
            dt64_nulls,
        }
    }
}
