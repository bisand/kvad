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
# started. Ask for those explicitly with --add-path and --service. A machine
# that already runs the service is the exception: there the quiet answer is to
# keep running it, on the binaries just installed.
#
# Upgrading keeps what the previous install decided: the service goes on
# listening on the address it was given, unless --host or --port says
# otherwise. It is also stopped before its binaries are replaced and started
# again afterwards, so nothing keeps serving from a file that has moved.
#
# Nothing here needs root. Nothing here writes outside $HOME unless you point
# --prefix somewhere else.

set -eu

REPO="bisand/kvad"
SERVICE_LABEL="net.kvad.serve"

# The address is carried as two fields and only joined where something needs
# the HOST:PORT spelling. One field at a time is harder to get wrong than one
# field with a colon in it, and when it is wrong, which half is wrong answers
# itself.
DEFAULT_HOST=127.0.0.1
DEFAULT_PORT=5823
SERVICE_HOST=$DEFAULT_HOST
SERVICE_PORT=$DEFAULT_PORT
HOST_GIVEN=0
PORT_GIVEN=0

# Set when the address was read off a service that is already installed,
# which changes what there is to ask: somebody upgrading answered this
# question the first time round. The host is tracked separately, because it
# is the half the warnings below are about.
BIND_FROM_UNIT=0
HOST_FROM_UNIT=0

# Set when an address has already been through the "would this actually
# start" questions, so they are not asked twice about the same address.
BIND_CHECKED=0

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

# The two halves joined for something that needs them as one string: a unit
# file, a URL, a message. IPv6 gets its brackets back here and nowhere else,
# so everything in between handles a bare host.
join_bind() { # host port
    case $1 in
        *:*) printf '[%s]:%s\n' "$1" "$2" ;;
        *)   printf '%s:%s\n' "$1" "$2" ;;
    esac
}

bind_addr() { join_bind "$SERVICE_HOST" "$SERVICE_PORT"; }

# A host this script is willing to write into a unit file. Not a hostname
# grammar — the server does the real parsing — just enough to catch what
# would otherwise become a unit that looks installed and cannot start.
#
# Reports rather than exits: a wrong flag should stop the run, but a typo at
# a prompt should only mean being asked again, and both need the same rules.
host_valid() {
    BIND_ERROR=""
    case $1 in
        '')            BIND_ERROR="the host is empty" ;;
        *://*)         BIND_ERROR="'$1' is a URL; the host on its own is enough, for example 127.0.0.1" ;;
        */*)           BIND_ERROR="'$1' is not a host; a path does not belong in it" ;;
        *[[:space:]]*) BIND_ERROR="'$1' has a space in it" ;;
        # Two colons or more is IPv6. Exactly one is a port that came along
        # when it was not asked for, and whose digits are not digits — the
        # prompt takes a well-formed HOST:PORT apart before it gets here.
        *:*:*) return 0 ;;
        *:*)   BIND_ERROR="'$1' has a colon in it but is not an IPv6 address; the port is a separate answer" ;;
        *) return 0 ;;
    esac
    return 1
}

# 1-65535. Port 0 is a real thing to bind — it means "any free port" — but
# a service at an address nobody can predict is not a service.
port_valid() {
    BIND_ERROR=""
    case $1 in
        ''|*[!0-9]*) BIND_ERROR="'$1' is not a port number"; return 1 ;;
    esac
    if [ "$1" -lt 1 ] || [ "$1" -gt 65535 ]; then
        BIND_ERROR="port $1 is outside 1-65535"
        return 1
    fi
    return 0
}

# HOST:PORT taken apart, for the places that still speak it: --bind, KVAD_BIND,
# and the ExecStart line of a unit this script wrote earlier. The port comes
# off the right and the brackets come off the host, so `[::1]:5823` divides
# where you would expect. Sets bind_host and bind_port.
split_bind() {
    BIND_ERROR=""
    case $1 in
        '['*']:'*) bind_port=${1##*:}; bind_host=${1%:*}
                   bind_host=${bind_host#\[}; bind_host=${bind_host%\]} ;;
        *:*)       bind_port=${1##*:}; bind_host=${1%:*} ;;
        *) BIND_ERROR="'$1' is not HOST:PORT, for example 127.0.0.1:5823"; return 1 ;;
    esac
    host_valid "$bind_host" && port_valid "$bind_port"
}

# What somebody typed at the host prompt: a host, or the whole HOST:PORT they
# have typed into address fields for twenty years. Taking that apart is
# friendlier than rejecting it, and the one ambiguous case — a bare IPv6
# literal, which is nothing but colons — is told apart by counting them:
# a single colon with digits to its right is a port, anything else is v6.
# Sets answer_host, and answer_port when one came along.
split_host_answer() {
    answer_host=$1
    answer_port=""
    case $1 in
        '['*']')   answer_host=${1#\[}; answer_host=${answer_host%\]} ;;
        '['*']:'*) answer_port=${1##*:}; answer_host=${1%:*}
                   answer_host=${answer_host#\[}; answer_host=${answer_host%\]} ;;
        *:*:*)     ;;
        *:[0-9]*)  answer_port=${1##*:}; answer_host=${1%:*} ;;
    esac
}

# HOST:PORT from --bind, which still means both halves at once.
set_bind() {
    split_bind "$1" || die "--bind: $BIND_ERROR"
    SERVICE_HOST=$bind_host; HOST_GIVEN=1
    SERVICE_PORT=$bind_port; PORT_GIVEN=1
}

is_loopback() {
    case $1 in
        127.*|localhost|::1|'[::1]') return 0 ;;
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
# The address matters, not just the port. A listener on `*:5823` does not
# stop a bind of `127.0.0.1:5823` — Rust sets SO_REUSEADDR, and an ssh
# forward on the wildcard happily coexists with a server on loopback. Asking
# "is this port in use anywhere" called that a conflict and was wrong.
address_taken() { # host port
    # Both tools spell an IPv6 address with brackets; everything else in
    # this script keeps the host bare, so they go back on here.
    case $1 in
        *:*) taken_host="[$1]" ;;
        *)   taken_host=$1 ;;
    esac
    if have lsof; then
        # shellcheck disable=SC2086 # the pid list is split on purpose
        pids=$(lsof -nP -iTCP@"$taken_host":"$2" -sTCP:LISTEN -t 2>/dev/null) || return 1
        [ -n "$pids" ] || return 1
        for pid in $pids; do
            # Our own service does not count. It is stopped before the
            # binaries are replaced, so this is mostly a server somebody
            # started by hand — but an agent that outlived its bootout is
            # still ours to take the address back from, and calling that a
            # conflict refused every upgrade, which is what it did.
            [ "$(exe_of "$pid")" = "$PREFIX/kvad-serve" ] || return 0
        done
        return 1
    elif have ss; then
        ss -ltnH 2>/dev/null | awk -v a="$taken_host:$2" '$4 == a { found = 1 } END { exit !found }'
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

# The two ways an address produces a unit that looks installed and never
# serves anything. Both are worth a question rather than a surprise.
# Returns 1 when the person would rather not.
bind_objections() { # 1 if the host is the one the service already uses
    addr=$(bind_addr)
    # A remembered host does not get the lecture. Somebody chose it, and an
    # auth mode it is allowed to bind under is already in the config file —
    # if it were not, the service would have been failing since long before
    # this upgrade came along.
    if [ "${1:-0}" -eq 0 ] && ! is_loopback "$SERVICE_HOST"; then
        say ""
        warn "$addr is not a loopback address, and kvad-serve refuses a
  non-loopback bind while auth.mode is \"none\" — which is the default. The
  service would fail to start and be restarted for as long as it is loaded.
  Set an auth mode first in ${XDG_CONFIG_HOME:-$HOME/.config}/kvad/kvad.toml;
  $PREFIX/kvad-serve --help and the bundled kvad.example.toml say how."
        ask "Install the service anyway?" n n || return 1
    fi
    if address_taken "$SERVICE_HOST" "$SERVICE_PORT"; then
        say ""
        warn "something is already listening on $addr itself. Two servers
  cannot share one address, so kvad-serve would fail to bind and be
  restarted in a loop. Pick another with --host or --port, or stop what is
  there."
        ask "Install the service anyway?" n n || return 1
    fi
    return 0
}

# Ask where the service should listen, and keep asking while the answer would
# produce one that cannot start. "Pick another with --port" is useless advice
# halfway through a run: the person is right here, so offer them the choice
# instead of sending them back to the shell.
#
# Only ever reached interactively, which is what makes the loop safe — every
# path out of it that is not `return` is a question somebody answered.
choose_bind() {
    while :; do
        [ "$INTERACTIVE" -eq 1 ] || return 1
        ask_value "Address for kvad-serve to listen on" "$SERVICE_HOST"
        split_host_answer "$VALUE"
        # Held as candidates until every check has passed. An address that
        # was just rejected must not become the default that Enter accepts,
        # or declining it walks straight back into the same warning.
        candidate_host=$answer_host
        if ! host_valid "$candidate_host"; then
            say "  $BIND_ERROR"
            continue
        fi
        if [ -n "$answer_port" ]; then
            candidate_port=$answer_port
            if ! port_valid "$candidate_port"; then
                say "  $BIND_ERROR"
                continue
            fi
            say "  ${DIM}port $candidate_port, from the address you typed${R}"
        else
            candidate_port=""
            while [ -z "$candidate_port" ]; do
                [ "$INTERACTIVE" -eq 1 ] || return 1
                ask_value "Port" "$SERVICE_PORT"
                if port_valid "$VALUE"; then
                    candidate_port=$VALUE
                else
                    say "  $BIND_ERROR"
                fi
            done
        fi
        candidate=$(join_bind "$candidate_host" "$candidate_port")
        if ! is_loopback "$candidate_host"; then
            warn "$candidate is not a loopback address, and kvad-serve refuses a
  non-loopback bind while auth.mode is \"none\" — which is the default. The
  service would fail to start and be restarted for as long as it is loaded.
  Set an auth mode first in ${XDG_CONFIG_HOME:-$HOME/.config}/kvad/kvad.toml;
  the bundled kvad.example.toml says how."
            ask "Use $candidate anyway?" n n || continue
        fi
        if address_taken "$candidate_host" "$candidate_port"; then
            warn "something is already listening on $candidate itself. Two servers
  cannot share one address, so kvad-serve would fail to bind and be
  restarted in a loop."
            ask "Use $candidate anyway?" n n || continue
        elif port_busy "$candidate_port"; then
            say "  ${DIM}note: something else is on port $candidate_port at another address."
            say "  That does not stop this one binding $candidate.${R}"
        fi
        SERVICE_HOST=$candidate_host
        SERVICE_PORT=$candidate_port
        BIND_CHECKED=1
        return 0
    done
}

# ------------------------------------------------------------- arguments --

PREFIX="${KVAD_INSTALL_DIR:-$HOME/.local/bin}"
VERSION="${KVAD_VERSION:-}"
WANT_SERVICE=ask
WANT_PATH=ask
UNINSTALL=0

# KVAD_BIND is the spelling this script started with, and somebody's second
# machine is still set up that way. It is read first, so KVAD_HOST and
# KVAD_PORT can override either half of it.
if [ -n "${KVAD_BIND:-}" ]; then
    split_bind "$KVAD_BIND" || die "KVAD_BIND: $BIND_ERROR"
    SERVICE_HOST=$bind_host; HOST_GIVEN=1
    SERVICE_PORT=$bind_port; PORT_GIVEN=1
fi
if [ -n "${KVAD_HOST:-}" ]; then
    SERVICE_HOST=$KVAD_HOST; HOST_GIVEN=1
    host_valid "$SERVICE_HOST" || die "KVAD_HOST: $BIND_ERROR"
fi
if [ -n "${KVAD_PORT:-}" ]; then
    SERVICE_PORT=$KVAD_PORT; PORT_GIVEN=1
    port_valid "$SERVICE_PORT" || die "KVAD_PORT: $BIND_ERROR"
fi

usage() {
    cat >&2 <<'USAGE'
install.sh — install kvad on macOS or Linux

    --prefix DIR     where the binaries go (default: ~/.local/bin)
    --version TAG    a release to install, e.g. v0.1.0 (default: the latest)
    --host ADDR      address the background service listens on
                     (default: 127.0.0.1, or the one it already listens on)
    --port PORT      the port it listens on (default: 5823, likewise)
    --bind HOST:PORT both at once, for a habit that is hard to break
    --service        install the background service without asking
    --no-service     skip it without asking
    --add-path       add the install directory to PATH without asking
    --no-add-path    leave PATH alone without asking
    -y, --yes        never ask; take the default answer to every question
    --uninstall      remove the binaries and the service, keep models and data
    -h, --help

Environment: KVAD_INSTALL_DIR, KVAD_VERSION, KVAD_HOST, KVAD_PORT, KVAD_BIND,
GITHUB_TOKEN (rate limits).
USAGE
    exit 2
}

# A malformed address should cost nothing to find out about, and the service
# section is on the far side of a 30 MB download. Checked as each one
# arrives, so the complaint names the flag that was wrong rather than the
# address the two of them added up to.
while [ $# -gt 0 ]; do
    case $1 in
        --host)        [ $# -ge 2 ] || die "--host needs an address"; SERVICE_HOST=$2; HOST_GIVEN=1
                       host_valid "$SERVICE_HOST" || die "--host: $BIND_ERROR"; shift 2 ;;
        --host=*)      SERVICE_HOST=${1#*=}; HOST_GIVEN=1
                       host_valid "$SERVICE_HOST" || die "--host: $BIND_ERROR"; shift ;;
        --port)        [ $# -ge 2 ] || die "--port needs a port number"; SERVICE_PORT=$2; PORT_GIVEN=1
                       port_valid "$SERVICE_PORT" || die "--port: $BIND_ERROR"; shift 2 ;;
        --port=*)      SERVICE_PORT=${1#*=}; PORT_GIVEN=1
                       port_valid "$SERVICE_PORT" || die "--port: $BIND_ERROR"; shift ;;
        --bind)        [ $# -ge 2 ] || die "--bind needs HOST:PORT"; set_bind "$2"; shift 2 ;;
        --bind=*)      set_bind "${1#*=}"; shift ;;
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

# Whether the service manager currently has the job, as opposed to there
# merely being a unit file on disk. The two come apart often enough —
# somebody stopped it by hand, a bootout that did not take — that guessing
# from the file is guessing.
service_loaded() {
    if [ "$PLATFORM" = macos ]; then
        launchctl print "gui/$(id -u)/$SERVICE_LABEL" >/dev/null 2>&1
    else
        have systemctl && systemctl --user is-active --quiet kvad-serve.service
    fi
}

# `launchctl bootout` returns while the job is still on its way out, and
# anything that touches the label in that gap fails. Wait for launchd to
# actually let go of it — five seconds, then give up and say so, because an
# install that hangs on a job that is never leaving is worse than one that
# tells you about it. Returns 1 if the label is still there.
await_unloaded() {
    waited=0
    while launchctl print "gui/$(id -u)/$SERVICE_LABEL" >/dev/null 2>&1; do
        waited=$((waited + 1))
        [ "$waited" -ge 50 ] && return 1
        sleep 0.1
    done
    return 0
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

# Stop a running service before the files under it move.
#
# A process keeps running from the binary it started with, so an install that
# only replaces files leaves the old version serving — on the address the
# new one is about to claim, and against a database the new one may have
# migrated on its way past. Out of the way first, back afterwards.
SERVICE_WAS_LOADED=0
suspend_service() {
    [ -f "$(service_paths)" ] || return 0
    service_loaded || return 0
    SERVICE_WAS_LOADED=1
    step "Stopping $SERVICE_LABEL before replacing its binaries"
    if [ "$PLATFORM" = macos ]; then
        launchctl bootout "gui/$(id -u)/$SERVICE_LABEL" >/dev/null 2>&1 || true
        await_unloaded || true
    else
        systemctl --user stop kvad-serve.service >/dev/null 2>&1 || true
    fi
    if service_loaded; then
        if [ "$PLATFORM" = macos ]; then
            how="launchctl bootout gui/$(id -u)/$SERVICE_LABEL"
        else
            how="systemctl --user stop kvad-serve"
        fi
        warn "$SERVICE_LABEL did not stop. The install continues, but the old
  server is still running from the binaries being replaced. Stop it with:
    $how"
    else
        say "  stopped"
    fi
}

# Put back what suspend_service took away, for the run that replaced the
# binaries and was not asked to rewrite the unit.
resume_service() {
    if [ "$PLATFORM" = macos ]; then
        launchctl bootstrap "gui/$(id -u)" "$(service_paths)" >/dev/null 2>&1 ||
            launchctl load -w "$(service_paths)" >/dev/null 2>&1 || true
    else
        systemctl --user start kvad-serve.service >/dev/null 2>&1 || true
    fi
    service_loaded
}

# The address a service that is already installed was given. An upgrade that
# quietly moved the server back to 127.0.0.1:5823 is a server that stopped
# answering where the rest of the machine expects it — so what is on disk is
# the default from here on. A flag still wins, and so does an answer at the
# prompt.
installed_bind() {
    unit=$(service_paths)
    [ -f "$unit" ] || return 1
    if [ "$PLATFORM" = macos ]; then
        # Every <string> in the file, one to a line — with grep rather than
        # a line-oriented sed, because a plist somebody reformatted by hand
        # can put the whole ProgramArguments array on one line and still be
        # the plist launchd is running. The address is the one after --bind.
        grep -o '<string>[^<]*</string>' "$unit" |
            sed 's|<string>\(.*\)</string>|\1|' |
            awk 'prev == "--bind" { print; exit } { prev = $0 }'
    else
        sed -n 's/^ExecStart=.*--bind[ =]\([^ ]*\).*/\1/p' "$unit" | head -n 1
    fi
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

# What the machine already decided, for the halves nobody named this time.
# Read before anything is downloaded, so that a run which never gets as far
# as the service section has still worked out what it would have said.
if [ "$HOST_GIVEN" -eq 0 ] || [ "$PORT_GIVEN" -eq 0 ]; then
    previous=$(installed_bind 2>/dev/null || true)
    if [ -n "$previous" ] && split_bind "$previous"; then
        [ "$HOST_GIVEN" -eq 1 ] || { SERVICE_HOST=$bind_host; HOST_FROM_UNIT=1; }
        [ "$PORT_GIVEN" -eq 1 ] || SERVICE_PORT=$bind_port
        BIND_FROM_UNIT=1
    elif [ -n "$previous" ]; then
        warn "the installed service listens on '$previous', which this script
  cannot make sense of. Falling back to $(bind_addr)."
    fi
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

mkdir -p "$PREFIX" || die "could not create $PREFIX"
[ -w "$PREFIX" ] || die "$PREFIX is not writable by this user.
  Pick somewhere else with --prefix DIR, or fix its permissions. This script
  deliberately does not use sudo."

# After the checks that can still refuse, and before the first byte moves:
# stopping somebody's server for an install that was going to fail on
# permissions anyway would be rude.
suspend_service

step "Installing into $PREFIX"
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
    addr=$(bind_addr)
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
        <string>$addr</string>
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
    domain="gui/$(id -u)"

    # bootout first, because bootstrap fails on a label that is already
    # there. Usually a no-op by now — suspend_service got there first — but
    # not when the service was loaded and not running, or when somebody
    # brought it back while this was downloading.
    if launchctl print "$domain/$SERVICE_LABEL" >/dev/null 2>&1; then
        launchctl bootout "$domain/$SERVICE_LABEL" >/dev/null 2>&1 || true
        await_unloaded || warn "the old $SERVICE_LABEL is still loaded after five seconds;
  loading the new one may fail. Boot it out yourself with:
    launchctl bootout $domain/$SERVICE_LABEL"
    fi

    launchctl bootstrap "$domain" "$unit" >/dev/null 2>&1 ||
        launchctl load -w "$unit" >/dev/null 2>&1 || true

    # Asked rather than inferred from an exit status, because `launchctl
    # load -w` prints "Load failed: 5: Input/output error" and then exits 0.
    # Trusting it meant reporting a loaded service while nothing was loaded
    # and nothing was listening.
    if launchctl print "$domain/$SERVICE_LABEL" >/dev/null 2>&1; then
        say "  loaded $SERVICE_LABEL"
        say "  logs: $logs/kvad-serve.log"
        say "  ${DIM}stop it with: launchctl bootout $domain/$SERVICE_LABEL${R}"
    else
        warn "wrote $unit, but launchd did not take it. Load it yourself with:
    launchctl bootstrap $domain $unit"
    fi
}

write_systemd() {
    unit=$(service_paths)
    addr=$(bind_addr)
    mkdir -p "$(dirname "$unit")"
    cat > "$unit" <<UNIT
[Unit]
Description=kvad — HTTP server and web UI
After=network.target

[Service]
ExecStart=$PREFIX/kvad-serve --bind $addr
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

if [ -f "$SRC/kvad-serve" ]; then
    do_service=0
    case $WANT_SERVICE in
        yes) do_service=1 ;;
        no)  do_service=0 ;;
        *)
            say ""
            if [ -f "$(service_paths)" ]; then
                # An upgrade, and the question is no longer whether to have a
                # service. Saying no here still puts the running one back;
                # saying yes rewrites the unit, which is how it picks up a
                # new --prefix or a new address.
                say "${B}kvad-serve${R} is already installed as a background service."
                ask "Update it for $VERSION?" y y && do_service=1
            else
                say "${B}kvad-serve${R} is the HTTP API and web UI."
                say "It can start automatically when you log in, or you can run it by hand."
                ask "Start kvad-serve at login?" n && do_service=1
            fi
            ;;
    esac

    # Anyone installing a service at a terminal gets asked where it listens,
    # including someone who passed --service. That flag answers "whether",
    # and saying yes to a service is not saying yes to port 5823.
    #
    # Two things skip the question. Naming either half with --host, --port
    # or --bind, which is an answer already — and an answer about one half
    # settles the other, which is whatever it was going to be anyway. And a
    # service that is already installed, which was asked once and has been
    # answering there ever since: an upgrade offers to keep its address
    # rather than asking again.
    if [ "$do_service" -eq 1 ] && [ "$INTERACTIVE" -eq 1 ] &&
       [ "$HOST_GIVEN" -eq 0 ] && [ "$PORT_GIVEN" -eq 0 ]; then
        if [ "$BIND_FROM_UNIT" -eq 1 ]; then
            say ""
            say "The installed service listens on ${B}$(bind_addr)${R}."
            ask "Keep that address?" y y || choose_bind || do_service=0
        else
            choose_bind || do_service=0
        fi
    fi

    # Every address that did not come out of choose_bind, which asks these as
    # it goes: a flag, an environment variable, a unit file kept as it was.
    if [ "$do_service" -eq 1 ] && [ "$BIND_CHECKED" -eq 0 ]; then
        bind_objections "$HOST_FROM_UNIT" || do_service=0
    fi

    if [ "$do_service" -eq 1 ]; then
        step "Installing the background service"
        if [ "$PLATFORM" = macos ]; then write_launchd; else write_systemd; fi
    elif [ "$SERVICE_WAS_LOADED" -eq 1 ]; then
        # Stopped a few steps up so its binaries could be replaced. Whatever
        # was just declined, it was not "turn my server off".
        step "Starting the service again on the new binaries"
        if resume_service; then
            say "  $SERVICE_LABEL is listening on $(bind_addr)"
        else
            if [ "$PLATFORM" = macos ]; then
                how="launchctl bootstrap gui/$(id -u) $(service_paths)"
            else
                how="systemctl --user start kvad-serve"
            fi
            warn "could not start $SERVICE_LABEL again. It is installed and stopped.
  Start it with:
    $how"
        fi
    elif [ -f "$(service_paths)" ]; then
        say "  ${DIM}$SERVICE_LABEL is installed but was not running; left that way.${R}"
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
if [ "$(bind_addr)" = "$DEFAULT_HOST:$DEFAULT_PORT" ]; then
    say "  ${B}kvad serve${R}                             the API and web UI on http://$(bind_addr)"
else
    say "  ${B}kvad serve --bind $(bind_addr)${R}   the API and web UI"
fi
say ""
say "Models go in ${XDG_DATA_HOME:-$HOME/.local/share}/kvad, configuration in"
say "${XDG_CONFIG_HOME:-$HOME/.config}/kvad. Uninstall with this script and --uninstall."
