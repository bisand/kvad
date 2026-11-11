-- A key made through a paired device is that device's, like the key pairing
-- gave it: it works only from that device through the tunnel, acts with the
-- device's role rather than the account's, and goes when the device does.
-- Without the column, `POST /api/keys` from a phone paired as `user` minted an
-- ordinary key of the account's: its full role, from anywhere.
--
-- NULL for a key made any other way, which is an ordinary key of the account.
ALTER TABLE api_keys ADD COLUMN device INTEGER REFERENCES devices(id) ON DELETE CASCADE;

-- The keys pairing made are their devices' already.
UPDATE api_keys SET device = (SELECT d.id FROM devices d WHERE d.api_key = api_keys.id);

CREATE INDEX api_keys_by_device ON api_keys (device);
