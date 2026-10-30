#!/bin/sh
# Kvad.app: the bundle the background service runs kvad-serve from on a Mac.
#
#     scripts/kvad-app.sh KVAD_SERVE VERSION DIR
#
# makes DIR/Kvad.app around a copy of KVAD_SERVE, unsigned. macOS lists a
# launch agent in Login Items under the app it belongs to, and one that
# belongs to none under the developer on its signature, so without the
# bundle the service shows as "Bogentech AS" rather than Kvad. The agent
# runs the copy in Contents/MacOS and names the bundle in
# AssociatedBundleIdentifiers (crates/llm/src/daemon.rs).
#
# LSBackgroundOnly: opened from Finder it starts a server with no window and
# no Dock icon, which is what it is. The icon is the iPhone app's.
set -eu
[ $# -eq 3 ] || { echo "usage: $0 KVAD_SERVE VERSION DIR" >&2; exit 2; }
serve=$1 version=${2#v} out=$3
root=$(cd "$(dirname "$0")/.." && pwd)
app="$out/Kvad.app"

rm -rf "$app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
cp "$serve" "$app/Contents/MacOS/kvad-serve"
chmod 755 "$app/Contents/MacOS/kvad-serve"

icon=$(mktemp -d)/Kvad.iconset
mkdir -p "$icon"
src="$root/apps/phone/ios/Assets.xcassets/AppIcon.appiconset/icon-1024.png"
for size in 16 32 128 256 512; do
    sips -z $size $size "$src" --out "$icon/icon_${size}x${size}.png" >/dev/null
    sips -z $((size * 2)) $((size * 2)) "$src" --out "$icon/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil -c icns "$icon" -o "$app/Contents/Resources/Kvad.icns"
rm -rf "$(dirname "$icon")"

cat > "$app/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key>         <string>eu.kvad.app</string>
    <key>CFBundleName</key>               <string>Kvad</string>
    <key>CFBundleDisplayName</key>        <string>Kvad</string>
    <key>CFBundleExecutable</key>         <string>kvad-serve</string>
    <key>CFBundleIconFile</key>           <string>Kvad</string>
    <key>CFBundlePackageType</key>        <string>APPL</string>
    <key>CFBundleShortVersionString</key> <string>$version</string>
    <key>CFBundleVersion</key>            <string>$version</string>
    <key>LSBackgroundOnly</key>           <true/>
</dict>
</plist>
PLIST
echo "$app"
