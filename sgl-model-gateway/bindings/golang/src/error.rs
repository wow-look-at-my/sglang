//! Error handling for FFI functions

use std::{ffi::CString, os::raw::c_char, ptr};

/// Error codes returned by FFI functions
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SglErrorCode {
    Success = 0,
    InvalidArgument = 1,
    TokenizationError = 2,
    ParsingError = 3,
    MemoryError = 4,
    UnknownError = 99,
}

/// Helper to set error message in FFI output parameter
///
/// # Safety
/// `error_out` must be null or point to a writable `*mut c_char`. A non-null
/// `message` leak is intentional: the caller frees it with `sgl_free_string`.
pub unsafe fn set_error_message(error_out: *mut *mut c_char, message: &str) {
    if !error_out.is_null() {
        if let Ok(cstr) = CString::new(message) {
            unsafe { *error_out = cstr.into_raw() };
        } else {
            unsafe { *error_out = ptr::null_mut() };
        }
    }
}

/// Helper to set error message from format string
///
/// # Safety
/// Inherits [`set_error_message`]'s requirement on `error_out`.
pub unsafe fn set_error_message_fmt(error_out: *mut *mut c_char, fmt: std::fmt::Arguments) {
    let msg = format!("{}", fmt);
    unsafe { set_error_message(error_out, &msg) };
}

/// Helper to clear error message
///
/// # Safety
/// `error_out` must be null or point to a writable `*mut c_char`. Any previous
/// value is leaked rather than freed, since ownership may already have moved
/// to the caller.
pub unsafe fn clear_error_message(error_out: *mut *mut c_char) {
    if !error_out.is_null() {
        unsafe { *error_out = ptr::null_mut() };
    }
}

// Helper functions for error handling
// Note: Some helper functions are kept for potential future use
