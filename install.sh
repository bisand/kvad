#!/bin/sh
#
# Install kvad from a GitHub release.
#
#     curl -fsSL https://raw.githubusercontent.com/bisand/kvad/master/install.sh | sh
#
# Downloads the release tarball for this machine, checks it against the
# release's SHA256SUMS, and puts the binaries in ~/.local/bin. Then it asks
# two questions — whether to put that directory on PATH, and whether
# kvad-serve should start at login — because the honest default for both is
# "it depends on the machine" and guessing wrong is worse than asking.
#
# It asks on /dev/tty rather than stdin, which is what lets the questions
# survive `curl | sh`: stdin there is the script itself.
#
# Non-interactive use (CI, Dockerfiles, a second machine):
#
#     curl -fsSL .../install.sh | sh -s -- --yes --service
#
# --yes never opens the terminal, and answers every question the quiet way:
# install the binaries and touch nothing else — no shell rc edited, no service
# started. Ask for those explicitly with --add-path and --service.
#
# Nothing here needs root. Nothing here writes outside $HOME unless you point
# --prefix somewhere else.

set -eu

REPO="bisand/kvad"
SERVICE_LABEL="net.kvad.serve"
SERVICE_BIND="${KVAD_BIND:-127.0.0.1:8080}"
BIND_GIVEN=0
[ -n "${KVAD_BIND:-}" ] && BIND_GIVEN=1

# ---------------------------------------------------------------- output --

# Colour only when stderr is a terminal. A log file full of escape codes is
# a worse log file, and `curl | sh 2>&1 | tee` is a thing people do.
if [ -t 2 ]; then
    B=$(printf '\033[1m'); DIM=$(printf '\033[2m'); RED=$(printf '\033[31m')
    YEL=$(printf '\033[33m'); GRN=$(printf '\033[32m'); R=$(printf '\033[0m')
else
    B=''; DIM=''; RED=''; YEL=''; GRN=''; R=''
fi

say()  { printf '%s\n' "$*" >&2; }
step() { printf '%s==>%s %s\n' "$B" "$R" "$*" >&2; }
warn() { printf '%swarning:%s %s\n' "$YEL" "$R" "$*" >&2; }
die()  { printf '%serror:%s %s\n' "$RED" "$R" "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

# ------------------------------------------------------------- questions --

# Interactive unless told otherwise, and only if there is a terminal to ask
# on. Both conditions matter: `--yes` is a choice, no /dev/tty is a fact.
#
# Opened rather than tested with -r: in a container started without a tty,
# /dev/tty exists and passes every permission check, and opening it fails
# with ENXIO. Asking is the only way to find out.
INTERACTIVE=1
{ : < /dev/tty; } 2>/dev/null || INTERACTIVE=0

# ask QUESTION DEFAULT [UNATTENDED]  ->  0 for yes, 1 for no
#
# DEFAULT is what Enter means, and UNATTENDED — if given — is what a run with
# --yes or no terminal gets instead. The two differ on purpose: "add this to
# your PATH?" is worth a yes from somebody reading the question, and is not
# something to do to a machine with nobody watching. Anything that is not a
# clear yes or no re-asks rather than guessing.
ask() {
    question=$1
    default=$2
    unattended=${3:-$2}
    case $default in
        y) hint="Y/n" ;;
        *) hint="y/N" ;;
    esac
    if [ "$INTERACTIVE" -eq 0 ]; then
        printf '%s %s[not asked -> %s]%s\n' "$question" "$DIM" "$unattended" "$R" >&2
        [ "$unattended" = y ]
        return
    fi
    while :; do
        printf '%s %s[%s]%s ' "$question" "$DIM" "$hint" "$R" > /dev/tty
        if ! read -r reply < /dev/tty; then
            # The terminal went away mid-question. Nobody is answering, so
            # this is an unattended run after all — and it takes the
            # unattended answer, not the one meant for somebody watching.
            # Latched, because a loop that re-asks would otherwise spin on
            # an EOF that is never going to turn into an answer.
            printf '\n' >&2
            INTERACTIVE=0
            [ "$unattended" = y ]
            return
        fi
        case $reply in
            "")            [ "$default" = y ]; return ;;
            y|Y|yes|YES)   return 0 ;;
            n|N|no|NO)     return 1 ;;
            *) printf '  please answer y or n\n' > /dev/tty ;;
        esac
    done
}

# ask_value QUESTION DEFAULT  ->  answer in VALUE
#
# Enter keeps the default, which is what makes this safe to put in front of
# somebody who does not care. An unattended run takes the default without
# asking, so --yes never blocks on a question nobody is there to answer.
ask_value() {
    VALUE=$2
    [ "$INTERACTIVE" -eq 1 ] || return 0
    printf '%s %s[%s]%s ' "$1" "$DIM" "$2" "$R" > /dev/tty
    if ! read -r reply < /dev/tty; then
        printf '\n' >&2
        INTERACTIVE=0
        return 0
    fi
    [ -n "$reply" ] && VALUE=$reply
    return 0
}

# ---------------------------------------------------------------- address --

# HOST:PORT, with IPv6 in brackets the way every other tool spells it. The
# port is split off the right so `[::1]:8080` divides where you would expect.
#
# Reports rather than exits: a wrong flag should stop the run, but a typo at a
# prompt should only mean being asked again, and both need the same rules.
# Sets bind_host and bind_port, which the checks below read.
bind_valid() {
    addr=$1
    BIND_ERROR=""
    case $addr in
        *:*) ;;
        *) BIND_ERROR="'$addr' is not HOST:PORT, for example 127.0.0.1:8080"; return 1 ;;
    esac
    bind_port=${addr##*:}
    bind_host=${addr%:*}
    case $bind_port in
        ''|*[!0-9]*) BIND_ERROR="'$bind_port' is not a port number"; return 1 ;;
    esac
    if [ "$bind_port" -lt 1 ] || [ "$bind_port" -gt 65535 ]; then
        BIND_ERROR="port $bind_port is outside 1-65535"
        return 1
    fi
    if [ -z "$bind_host" ]; then
        BIND_ERROR="no host in '$addr'"
        return 1
    fi
    return 0
}

# The same rules, for an address that came from a flag: there is nobody to
# ask again, so a bad one ends the run.
check_bind() {
    bind_valid "$1" || die "--bind: $BIND_ERROR"
}

is_loopback() {
    case $1 in
        127.*|localhost|'[::1]'|::1) return 0 ;;
        *) return 1 ;;
    esac
}

# What a pid is running, as far as each platform will say. Linux's `ps
# -o comm=` gives a short name, so /proc is asked first where it exists;
# macOS has no /proc and answers with the full path.
exe_of() { # pid
    if [ -r "/proc/$1/exe" ]; then
        readlink "/proc/$1/exe" 2>/dev/null
    else
        ps -o comm= -p "$1" 2>/dev/null
    fi
}

# Whether something already holds this exact address. Worth knowing before
# installing a unit with KeepAlive on it: kvad-serve would fail to bind, be
# restarted, fail again, and do that forever while looking installed.
#
# The address matters, not just the port. A listener on `*:8080` does not
# stop a bind of `127.0.0.1:8080` — Rust sets SO_REUSEADDR, and an ssh
# forward on the wildcard happily coexists with a server on loopback. Asking
# "is this port in use anywhere" called that a conflict and was wrong.
address_taken() { # host port
    if have lsof; then
        # shellcheck disable=SC2086 # the pid list is split on purpose
        pids=$(lsof -nP -iTCP@"$1":"$2" -sTCP:LISTEN -t 2>/dev/null) || return 1
        [ -n "$pids" ] || return 1
        for pid in $pids; do
            # Our own service does not count. Upgrading in place leaves the
            # old agent listening on the very address the new one wants, and
            # installing the service boots it out before bootstrapping the
            # replacement -- so the address is ours to take back. Calling it
            # a conflict would refuse every upgrade, which is what it did.
            [ "$(exe_of "$pid")" = "$PREFIX/kvad-serve" ] || return 0
        done
        return 1
    elif have ss; then
        ss -ltnH 2>/dev/null | awk -v a="$1:$2" '$4 == a { found = 1 } END { exit !found }'
    else
        return 1
    fi
}

# Anything at all on the port, whatever address it is bound to. Not a
# conflict, but worth mentioning: it is why a server can come up and still
# not be the thing answering on someone else's interface.
port_busy() {
    if have lsof; then
        lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
    elif have ss; then
        ss -ltnH 2>/dev/null | awk '{print $4}' | grep -q "[:.]$1$"
    else
        return 1
    fi
}

# Ask where the service should listen, and keep asking while the answer would
# produce one that cannot start. "Pick another with --bind" is useless advice
# halfway through a run: the person is right here, so offer them the choice
# instead of sending them back to the shell.
#
# Only ever reached interactively, which is what makes the loop safe — every
# path out of it that is not `return` is a question somebody answered.
choose_bind() {
    while :; do
        [ "$INTERACTIVE" -eq 1 ] || return 1
        ask_value "Address for kvad-serve to listen on" "$SERVICE_BIND"
        # Held as a candidate until every check has passed. An address that
        # was just rejected must not become the default that Enter accepts,
        # or declining it walks straight back into the same warning.
        candidate=$VALUE
        if ! bind_valid "$candidate"; then
            say "  $BIND_ERROR"
            continue
        fi
        if ! is_loopback "$bind_host"; then
            warn "$candidate is not a loopback address, and kvad-serve refuses a
  non-loopback bind while auth.mode is \"none\" — which is the default. The
  service would fail to start and be restarted for as long as it is loaded.
  Set an auth mode first in ${XDG_CONFIG_HOME:-$HOME/.config}/kvad/kvad.toml;
  the bundled kvad.example.toml says how."
            ask "Use $candidate anyway?" n n || continue
        fi
        if address_taken "$bind_host" "$bind_port"; then
            warn "something is already listening on $candidate itself. Two servers
  cannot share one address, so kvad-serve would fail to bind and be
  restarted in a loop."
            ask "Use $candidate anyway?" n n || continue
        elif port_busy "$bind_port"; then
            say "  ${DIM}note: something else is on port $bind_port at another address."
            say "  That does not stop this one binding $candidate.${R}"
        fi
        SERVICE_BIND=$candidate
        return 0
    done
}

# ------------------------------------------------------------- arguments --

PREFIX="${KVAD_INSTALL_DIR:-$HOME/.local/bin}"
VERSION="${KVAD_VERSION:-}"
WANT_SERVICE=ask
WANT_PATH=ask
UNINSTALL=0

usage() {
    cat >&2 <<'USAGE'
install.sh — install kvad on macOS or Linux

    --prefix DIR     where the binaries go (default: ~/.local/bin)
    --version TAG    a release to install, e.g. v0.1.0 (default: the latest)
    --bind ADDR      address the background service listens on
                     (default: 127.0.0.1:8080)
    --service        install the background service without asking
    --no-service     skip it without asking
    --add-path       add the install directory to PATH without asking
    --no-add-path    leave PATH alone without asking
    -y, --yes        never ask; take the default answer to every question
    --uninstall      remove the binaries and the service, keep models and data
    -h, --help

Environment: KVAD_INSTALL_DIR, KVAD_VERSION, KVAD_BIND, GITHUB_TOKEN (rate limits).
USAGE
    exit 2
}

while [ $# -gt 0 ]; do
    case $1 in
        --bind)        [ $# -ge 2 ] || die "--bind needs HOST:PORT"; SERVICE_BIND=$2; BIND_GIVEN=1; shift 2 ;;
        --bind=*)      SERVICE_BIND=${1#*=}; BIND_GIVEN=1; shift ;;
        --prefix)      [ $# -ge 2 ] || die "--prefix needs a directory"; PREFIX=$2; shift 2 ;;
        --version)     [ $# -ge 2 ] || die "--version needs a tag"; VERSION=$2; shift 2 ;;
        --prefix=*)    PREFIX=${1#*=}; shift ;;
        --version=*)   VERSION=${1#*=}; shift ;;
        --service)     WANT_SERVICE=yes; shift ;;
        --no-service)  WANT_SERVICE=no; shift ;;
        --add-path)    WANT_PATH=yes; shift ;;
        --no-add-path) WANT_PATH=no; shift ;;
        -y|--yes)      INTERACTIVE=0; shift ;;
        --uninstall)   UNINSTALL=1; shift ;;
        -h|--help)     usage ;;
        *)             say "unknown option: $1"; usage ;;
    esac
done

# A malformed address should cost nothing to find out about, and the service
# section is on the far side of a 30 MB download.
check_bind "$SERVICE_BIND"

# ---------------------------------------------------------------- machine --

# The triple is both the artifact name and the answer to "is this machine
# supported", so getting it wrong should stop here rather than 404 later.
detect_target() {
    os=$(uname -s)
    arch=$(uname -m)
    case $os in
        Darwin) os=apple-darwin ;;
        Linux)  os=unknown-linux-gnu ;;
        *) die "unsupported operating system: $os (this installs on macOS and Linux)" ;;
    esac
    case $arch in
        arm64|aarch64) arch=aarch64 ;;
        x86_64|amd64)  arch=x86_64 ;;
        *) die "unsupported architecture: $arch" ;;
    esac
    printf '%s-%s\n' "$arch" "$os"
}

TARGET=$(detect_target)
case $TARGET in
    *apple-darwin) PLATFORM=macos ;;
    *)             PLATFORM=linux ;;
esac

# ------------------------------------------------------------- uninstall --

service_paths() {
    if [ "$PLATFORM" = macos ]; then
        printf '%s\n' "$HOME/Library/LaunchAgents/$SERVICE_LABEL.plist"
    else
        printf '%s\n' "${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/kvad-serve.service"
    fi
}

stop_service() {
    unit=$(service_paths)
    [ -f "$unit" ] || return 0
    step "Stopping the service"
    if [ "$PLATFORM" = macos ]; then
        launchctl bootout "gui/$(id -u)/$SERVICE_LABEL" 2>/dev/null ||
            launchctl unload -w "$unit" 2>/dev/null || true
    else
        systemctl --user disable --now kvad-serve.service 2>/dev/null || true
    fi
    rm -f "$unit"
    say "  removed $unit"
}

if [ "$UNINSTALL" -eq 1 ]; then
    stop_service
    step "Removing binaries from $PREFIX"
    removed=0
    for name in kvad kvad-serve kvad-tui kvad-gpu; do
        if [ -e "$PREFIX/$name" ]; then
            rm -f "$PREFIX/$name"
            say "  removed $PREFIX/$name"
            removed=$((removed + 1))
        fi
    done
    [ "$removed" -gt 0 ] || say "  nothing to remove"
    say ""
    say "Models, conversations and configuration were left alone. They are in:"
    say "  ${XDG_DATA_HOME:-$HOME/.local/share}/kvad"
    say "  ${XDG_CONFIG_HOME:-$HOME/.config}/kvad"
    say "Delete those directories to remove them too."
    exit 0
fi

# -------------------------------------------------------------- download --

have curl || have wget || die "need curl or wget to download anything"
have tar  || die "need tar to unpack the release"

# GITHUB_TOKEN is spelled out in both branches rather than built into a
# variable: an expansion holding `-H "Authorization: ..."` would be split on
# its spaces, not on its quotes, and the header would arrive in pieces.
fetch() { # fetch URL OUTPUT
    if have curl; then
        if [ -n "${GITHUB_TOKEN:-}" ]; then
            curl -fsSL -H "Authorization: Bearer $GITHUB_TOKEN" -o "$2" "$1"
        else
            curl -fsSL -o "$2" "$1"
        fi
    else
        if [ -n "${GITHUB_TOKEN:-}" ]; then
            wget -q --header="Authorization: Bearer $GITHUB_TOKEN" -O "$2" "$1"
        else
            wget -qO "$2" "$1"
        fi
    fi
}

fetch_stdout() { # fetch URL, to stdout
    if have curl; then
        if [ -n "${GITHUB_TOKEN:-}" ]; then
            curl -fsSL -H "Authorization: Bearer $GITHUB_TOKEN" "$1"
        else
            curl -fsSL "$1"
        fi
    else
        if [ -n "${GITHUB_TOKEN:-}" ]; then
            wget -q --header="Authorization: Bearer $GITHUB_TOKEN" -O- "$1"
        else
            wget -qO- "$1"
        fi
    fi
}

if [ -z "$VERSION" ]; then
    step "Finding the latest release"
    # Parsed with sed rather than jq, which is not on a stock macOS. The
    # field is the release's tag, and it is the first "tag_name" in the JSON.
    VERSION=$(fetch_stdout "https://api.github.com/repos/$REPO/releases/latest" 2>/dev/null |
        sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -n 1) || true
    [ -n "$VERSION" ] || die "could not find a release of $REPO.
  GitHub's API may be rate-limiting this machine — set GITHUB_TOKEN, or pick
  a version yourself with --version v0.1.0. Releases are listed at
  https://github.com/$REPO/releases"
fi

NAME="kvad-$VERSION-$TARGET"
BASE="https://github.com/$REPO/releases/download/$VERSION"

TMP=$(mktemp -d "${TMPDIR:-/tmp}/kvad-install.XXXXXX")
trap 'rm -rf "$TMP"' EXIT INT TERM

step "Downloading $NAME.tar.gz"
fetch "$BASE/$NAME.tar.gz" "$TMP/$NAME.tar.gz" || die "no build of $VERSION for $TARGET.
  Available downloads: https://github.com/$REPO/releases/tag/$VERSION"

# A checksum you fetch from the same place as the file is not a signature,
# and is not claimed to be one. What it does catch is a truncated download
# and a corrupted mirror, which are the failures that actually happen here.
step "Checking the download against SHA256SUMS"
if fetch "$BASE/SHA256SUMS" "$TMP/SHA256SUMS" 2>/dev/null; then
    # Compared as a whole field, not matched as a pattern: the name is full
    # of dots, and a regex would take them for wildcards.
    expected=$(awk -v want="$NAME.tar.gz" '$2 == want { print $1; exit }' "$TMP/SHA256SUMS")
    if [ -z "$expected" ]; then
        warn "SHA256SUMS does not mention $NAME.tar.gz; continuing unverified"
    else
        if have sha256sum; then
            actual=$(sha256sum "$TMP/$NAME.tar.gz" | awk '{print $1}')
        elif have shasum; then
            actual=$(shasum -a 256 "$TMP/$NAME.tar.gz" | awk '{print $1}')
        else
            actual=$expected
            warn "no sha256sum or shasum on this machine; continuing unverified"
        fi
        [ "$actual" = "$expected" ] || die "checksum mismatch for $NAME.tar.gz
  expected $expected
  got      $actual
  Do not use this download."
        say "  ${GRN}ok${R} $expected"
    fi
else
    warn "no SHA256SUMS in release $VERSION; continuing unverified"
fi

tar -xzf "$TMP/$NAME.tar.gz" -C "$TMP"
SRC="$TMP/$NAME"
[ -d "$SRC" ] || die "the archive did not contain a $NAME directory"

# -------------------------------------------------------------- install ---

# Whatever the tarball holds, rather than a fixed list: a Linux build has no
# kvad-tui or kvad-gpu, because both link Metal.
BINARIES=""
for candidate in kvad kvad-serve kvad-tui kvad-gpu; do
    [ -f "$SRC/$candidate" ] && BINARIES="$BINARIES $candidate"
done
[ -n "$BINARIES" ] || die "the archive contained no binaries"

if [ -e "$PREFIX/kvad" ]; then
    current=$("$PREFIX/kvad" --version 2>/dev/null || echo "an unknown version")
    say ""
    say "$PREFIX already has $current installed."
    ask "Replace it with $VERSION?" y || { say "Left alone."; exit 0; }
fi

step "Installing into $PREFIX"
mkdir -p "$PREFIX" || die "could not create $PREFIX"
[ -w "$PREFIX" ] || die "$PREFIX is not writable by this user.
  Pick somewhere else with --prefix DIR, or fix its permissions. This script
  deliberately does not use sudo."

for name in $BINARIES; do
    # Install to a temporary name and rename, so a running kvad-serve is
    # replaced atomically instead of being overwritten under its own feet.
    install_tmp="$PREFIX/.$name.new.$$"
    cp "$SRC/$name" "$install_tmp"
    chmod 755 "$install_tmp"
    mv -f "$install_tmp" "$PREFIX/$name"
    say "  $PREFIX/$name"
done

for extra in LICENSE README.md kvad.example.toml; do
    [ -f "$SRC/$extra" ] || continue
    doc="${XDG_DATA_HOME:-$HOME/.local/share}/kvad/doc"
    mkdir -p "$doc"
    cp "$SRC/$extra" "$doc/$extra"
done

# Downloads made by curl are not quarantined — Gatekeeper's flag comes from
# LaunchServices, so a browser sets it and this does not. Cleared anyway, for
# the person who downloaded the tarball by hand and then ran this on it.
if [ "$PLATFORM" = macos ] && have xattr; then
    for name in $BINARIES; do
        xattr -d com.apple.quarantine "$PREFIX/$name" 2>/dev/null || true
    done
fi

installed=$("$PREFIX/kvad" --version 2>/dev/null || true)
[ -n "$installed" ] && say "  ${GRN}ok${R} $installed runs on this machine"

# ------------------------------------------------------------------ PATH --

on_path() {
    case ":${PATH:-}:" in
        *":$PREFIX:"*) return 0 ;;
        *) return 1 ;;
    esac
}

rc_file() {
    # The file the user's *login* shell reads, which is not necessarily the
    # shell running this script — `curl | sh` is always /bin/sh.
    shell=$(basename "${SHELL:-/bin/sh}")
    case $shell in
        zsh)  printf '%s\n' "${ZDOTDIR:-$HOME}/.zshrc" ;;
        bash) if [ "$PLATFORM" = macos ] && [ -f "$HOME/.bash_profile" ]; then
                  printf '%s\n' "$HOME/.bash_profile"
              else
                  printf '%s\n' "$HOME/.bashrc"
              fi ;;
        fish) printf '%s\n' "${XDG_CONFIG_HOME:-$HOME/.config}/fish/config.fish" ;;
        *)    printf '%s\n' "$HOME/.profile" ;;
    esac
}

if ! on_path; then
    RC=$(rc_file)
    say ""
    say "$PREFIX is not on your PATH, so ${B}kvad${R} will not be found by name."
    do_path=0
    case $WANT_PATH in
        yes) do_path=1 ;;
        no)  do_path=0 ;;
        *)   ask "Add it to $RC?" y n && do_path=1 ;;
    esac
    if [ "$do_path" -eq 1 ]; then
        mkdir -p "$(dirname "$RC")"
        if [ -f "$RC" ] && grep -q 'added by kvad install.sh' "$RC"; then
            say "  $RC already has the line"
        else
            case $RC in
                *config.fish) printf '\n# added by kvad install.sh\nfish_add_path %s\n' "$PREFIX" >> "$RC" ;;
                *)            printf '\n# added by kvad install.sh\nexport PATH="%s:$PATH"\n' "$PREFIX" >> "$RC" ;;
            esac
            say "  added to $RC"
        fi
        say "  ${DIM}open a new terminal, or run: export PATH=\"$PREFIX:\$PATH\"${R}"
    elif [ "$INTERACTIVE" -eq 0 ]; then
        say "  ${DIM}pass --add-path to have this script add it${R}"
    else
        say "  ${DIM}run it as $PREFIX/kvad, or add that directory to PATH yourself${R}"
    fi
fi

# --------------------------------------------------------------- service --

write_launchd() {
    unit=$(service_paths)
    logs="$HOME/Library/Logs/kvad"
    mkdir -p "$(dirname "$unit")" "$logs"
    cat > "$unit" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>              <string>$SERVICE_LABEL</string>
    <key>ProgramArguments</key>
    <array>
        <string>$PREFIX/kvad-serve</string>
        <string>--bind</string>
        <string>$SERVICE_BIND</string>
    </array>
    <key>RunAtLoad</key>          <true/>
    <key>KeepAlive</key>
    <dict><key>SuccessfulExit</key><false/></dict>
    <key>ProcessType</key>        <string>Interactive</string>
    <key>StandardOutPath</key>    <string>$logs/kvad-serve.log</string>
    <key>StandardErrorPath</key>  <string>$logs/kvad-serve.err</string>
    <key>WorkingDirectory</key>   <string>$HOME</string>
</dict>
</plist>
PLIST
    # bootout first: bootstrap fails on an already-loaded label, and an
    # upgrade is the common case here.
    launchctl bootout "gui/$(id -u)/$SERVICE_LABEL" 2>/dev/null || true
    if launchctl bootstrap "gui/$(id -u)" "$unit" 2>/dev/null ||
       launchctl load -w "$unit" 2>/dev/null; then
        say "  loaded $SERVICE_LABEL"
        say "  logs: $logs/kvad-serve.log"
        say "  ${DIM}stop it with: launchctl bootout gui/$(id -u)/$SERVICE_LABEL${R}"
    else
        warn "wrote $unit but launchctl would not load it; load it yourself with:
    launchctl bootstrap gui/$(id -u) $unit"
    fi
}

write_systemd() {
    unit=$(service_paths)
    mkdir -p "$(dirname "$unit")"
    cat > "$unit" <<UNIT
[Unit]
Description=kvad — HTTP server and web UI
After=network.target

[Service]
ExecStart=$PREFIX/kvad-serve --bind $SERVICE_BIND
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
UNIT
    if have systemctl && systemctl --user daemon-reload 2>/dev/null; then
        systemctl --user enable --now kvad-serve.service
        say "  enabled kvad-serve.service"
        say "  logs: journalctl --user -u kvad-serve -f"
        say "  ${DIM}stop it with: systemctl --user disable --now kvad-serve${R}"
        # A user service dies with the last session unless lingering is on,
        # which is surprising enough on a headless box to be worth saying.
        if have loginctl && [ "$(loginctl show-user "$(id -un)" -p Linger --value 2>/dev/null || echo no)" != yes ]; then
            say "  ${DIM}to keep it running after you log out: sudo loginctl enable-linger $(id -un)${R}"
        fi
    else
        warn "wrote $unit but systemd --user is not available here; start the
  server yourself with: $PREFIX/kvad-serve"
    fi
}

# Kick a service that is already installed, so it runs the binary that is
# now on disk.
restart_service() {
    if [ "$PLATFORM" = macos ]; then
        launchctl kickstart -k "gui/$(id -u)/$SERVICE_LABEL" >/dev/null 2>&1
    else
        systemctl --user restart kvad-serve.service >/dev/null 2>&1
    fi
}

if [ -f "$SRC/kvad-serve" ]; then
    do_service=0
    case $WANT_SERVICE in
        yes) do_service=1 ;;
        no)  do_service=0 ;;
        *)
            say ""
            say "${B}kvad-serve${R} is the HTTP API and web UI."
            say "It can start automatically when you log in, or you can run it by hand."
            ask "Start kvad-serve at login?" n && do_service=1
            ;;
    esac

    # Anyone installing a service at a terminal gets asked where it listens,
    # including someone who passed --service. That flag answers "whether",
    # and saying yes to a service is not saying yes to port 8080. Only
    # --bind, which names an address, skips the question.
    if [ "$do_service" -eq 1 ] && [ "$BIND_GIVEN" -eq 0 ] && [ "$INTERACTIVE" -eq 1 ]; then
        choose_bind || do_service=0
    elif [ "$do_service" -eq 1 ]; then
        # Both of these produce a unit that looks installed and never serves
        # anything, so they are worth a question rather than a surprise.
        if ! is_loopback "$bind_host"; then
            say ""
            warn "$SERVICE_BIND is not a loopback address, and kvad-serve refuses a
  non-loopback bind while auth.mode is \"none\" — which is the default. The
  service would fail to start and be restarted for as long as it is loaded.
  Set an auth mode first in ${XDG_CONFIG_HOME:-$HOME/.config}/kvad/kvad.toml;
  $PREFIX/kvad-serve --help and the bundled kvad.example.toml say how."
            ask "Install the service anyway?" n n || do_service=0
        fi
        if [ "$do_service" -eq 1 ] && address_taken "$bind_host" "$bind_port"; then
            say ""
            warn "something is already listening on $SERVICE_BIND itself. Two servers
  cannot share one address, so kvad-serve would fail to bind and be
  restarted in a loop. Pick another with --bind, or stop what is there."
            ask "Install the service anyway?" n n || do_service=0
        fi
    fi
    if [ "$do_service" -eq 1 ]; then
        step "Installing the background service"
        if [ "$PLATFORM" = macos ]; then write_launchd; else write_systemd; fi
    elif [ -f "$(service_paths)" ]; then
        # There is a service, and the binary underneath it has just been
        # replaced. A running process keeps the file it started from, so
        # without this the install finishes, reports the new version, and
        # leaves the old one serving -- which is the sort of thing somebody
        # only discovers when a bug they read the fix for is still there.
        step "Restarting the service onto the new binaries"
        if restart_service; then
            say "  restarted $SERVICE_LABEL"
        else
            if [ "$PLATFORM" = macos ]; then
                how="launchctl kickstart -k gui/$(id -u)/$SERVICE_LABEL"
            else
                how="systemctl --user restart kvad-serve"
            fi
            warn "could not restart $SERVICE_LABEL. It is still running the binary it
  started with, not the one just installed. Restart it with:
    $how"
        fi
    fi
fi

# ------------------------------------------------------------------ done --

say ""
say "${GRN}kvad $VERSION is installed.${R}"
say ""
say "  ${B}kvad pull${R} Qwen/Qwen2.5-0.5B-Instruct   download a model"
say "  ${B}kvad chat${R}                              talk to it"
if [ -f "$SRC/kvad-tui" ]; then
    say "  ${B}kvad-tui${R}                               browse and chat in the terminal"
fi
if [ "$BIND_GIVEN" -eq 1 ]; then
    say "  ${B}kvad serve --bind $SERVICE_BIND${R}   the API and web UI"
else
    say "  ${B}kvad serve${R}                             the API and web UI on http://$SERVICE_BIND"
fi
say ""
say "Models go in ${XDG_DATA_HOME:-$HOME/.local/share}/kvad, configuration in"
say "${XDG_CONFIG_HOME:-$HOME/.config}/kvad. Uninstall with this script and --uninstall."
