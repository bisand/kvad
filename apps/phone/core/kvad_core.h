// Kvad for iPhone's Rust core, as the SwiftUI app calls it: three functions,
// and one JSON object a message each way. Every command and event is in
// ../messages.json. docs/phone-swiftui-plan.md has the design.

#ifndef KVAD_CORE_H
#define KVAD_CORE_H

// Starts the core, once; later calls change nothing. Its first event is
// `ready`. `on_event` is called on any of the core's threads with `context`
// and a NUL-terminated JSON object, which the core frees when the call
// returns: copy it, then hop to the main thread.
void kvad_core_start(void *context, void (*on_event)(void *context, const char *json));

// One command, as a NUL-terminated JSON object. It returns at once; what
// comes of it are events.
void kvad_core_send(const char *json);

// The core's version, for the About screen. Static; never freed.
const char *kvad_core_version(void);

#endif
