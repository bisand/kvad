// Every event in apps/phone/messages.json decodes into an Event this app
// knows, as the core's tests check that every event it sends is listed
// there. Between them, a message renamed on one side fails a test.

import XCTest
@testable import Kvad

final class MessagesTests: XCTestCase {
    func testEveryListedEventDecodes() throws {
        let url = try XCTUnwrap(Bundle(for: Self.self).url(forResource: "messages", withExtension: "json"))
        let all = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        let events = try XCTUnwrap(all["events"] as? [String: Any])
        XCTAssertFalse(events.isEmpty)
        for (name, body) in events {
            let data = try JSONSerialization.data(withJSONObject: [name: body])
            let event = try JSONDecoder().decode(Event.self, from: data)
            if case .unknown = event {
                XCTFail("\(name) is in messages.json, and Event does not know it")
            }
        }
    }

    func testAnEventFromANewerCoreIsUnknownNotAnError() throws {
        let event = try JSONDecoder().decode(Event.self, from: Data(#"{"something_new": {"x": 1}}"#.utf8))
        guard case .unknown("something_new") = event else { return XCTFail("\(event)") }
    }
}
