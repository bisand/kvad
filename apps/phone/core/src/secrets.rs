//! The one Keychain item this app keeps: its pairings, as JSON, with the
//! device key in them.
//!
//! Accessible only while the phone is unlocked, on this phone only: never
//! synced to iCloud, never in a backup that another phone could restore
//! (`kSecAttrAccessibleWhenUnlockedThisDeviceOnly`). On the Mac, where the
//! UI is iterated on, it is kept in memory instead, so that a development
//! window never writes to the login keychain.

pub type Res<T> = Result<T, String>;

#[cfg(target_os = "ios")]
mod store {
    use security_framework::access_control::{ProtectionMode, SecAccessControl};
    use security_framework::passwords::{delete_generic_password, generic_password, set_generic_password_options};
    use security_framework::passwords_options::PasswordOptions;

    const SERVICE: &str = "eu.kvad.phone";
    const ACCOUNT: &str = "pairings";

    pub fn get() -> Option<Vec<u8>> {
        generic_password(PasswordOptions::new_generic_password(SERVICE, ACCOUNT)).ok()
    }

    pub fn set(bytes: &[u8]) -> super::Res<()> {
        // Replaced rather than updated, so that the item is always made with
        // the protection below, whatever an older build made it with.
        let _ = delete_generic_password(SERVICE, ACCOUNT);
        let mut options = PasswordOptions::new_generic_password(SERVICE, ACCOUNT);
        let access = SecAccessControl::create_with_protection(Some(ProtectionMode::AccessibleWhenUnlockedThisDeviceOnly), 0)
            .map_err(|e| format!("the Keychain: {e}"))?;
        options.set_access_control(access);
        options.set_access_synchronized(Some(false));
        set_generic_password_options(bytes, options).map_err(|e| format!("the Keychain: {e}"))
    }
}

#[cfg(not(target_os = "ios"))]
mod store {
    static KEPT: std::sync::Mutex<Option<Vec<u8>>> = std::sync::Mutex::new(None);

    pub fn get() -> Option<Vec<u8>> {
        KEPT.lock().unwrap().clone()
    }

    pub fn set(bytes: &[u8]) -> super::Res<()> {
        *KEPT.lock().unwrap() = Some(bytes.to_vec());
        Ok(())
    }
}

pub fn get() -> Option<Vec<u8>> {
    store::get()
}

pub fn set(bytes: &[u8]) -> Res<()> {
    store::set(bytes)
}
