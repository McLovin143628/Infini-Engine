// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import {
  __resetTourForTest,
  TOUR_STEPS,
  tourSeen,
  useTourStore,
} from "../tourStore";

const tour = () => useTourStore.getState();

describe("tourStore state machine", () => {
  beforeEach(() => {
    localStorage.clear();
    __resetTourForTest();
  });
  afterEach(() => {
    __resetTourForTest();
  });

  it("starts inactive and unseen", () => {
    expect(tour().active).toBe(false);
    expect(tour().step).toBe(0);
    expect(tourSeen()).toBe(false);
  });

  it("start() activates at step 0", () => {
    tour().start();
    expect(tour().active).toBe(true);
    expect(tour().step).toBe(0);
  });

  it("next() advances through the steps", () => {
    tour().start();
    tour().next();
    expect(tour().step).toBe(1);
    tour().next();
    expect(tour().step).toBe(2);
  });

  it("prev() clamps at 0", () => {
    tour().start();
    tour().prev();
    expect(tour().step).toBe(0);
    tour().next();
    tour().prev();
    expect(tour().step).toBe(0);
  });

  it("next() on the last step finishes and marks seen", () => {
    tour().start();
    // Advance to the final step.
    for (let i = 0; i < TOUR_STEPS.length - 1; i++) tour().next();
    expect(tour().step).toBe(TOUR_STEPS.length - 1);
    expect(tour().active).toBe(true);
    // One more Next completes the tour.
    tour().next();
    expect(tour().active).toBe(false);
    expect(tour().step).toBe(0);
    expect(tourSeen()).toBe(true);
  });

  it("skip() dismisses and marks seen", () => {
    tour().start();
    tour().next();
    tour().skip();
    expect(tour().active).toBe(false);
    expect(tourSeen()).toBe(true);
  });

  it("finish() marks seen and resets", () => {
    tour().start();
    tour().finish();
    expect(tour().active).toBe(false);
    expect(tour().step).toBe(0);
    expect(tourSeen()).toBe(true);
  });

  it("maybeAutostart() runs once, then never again after seen", () => {
    // First-run: unseen → autostarts.
    tour().maybeAutostart();
    expect(tour().active).toBe(true);
    // Dismiss it.
    tour().skip();
    expect(tourSeen()).toBe(true);
    // Subsequent autostart attempts are no-ops.
    tour().maybeAutostart();
    expect(tour().active).toBe(false);
  });

  it("maybeAutostart() is a no-op while already active", () => {
    tour().start();
    tour().next();
    tour().maybeAutostart();
    // Did not reset back to step 0.
    expect(tour().step).toBe(1);
  });

  it("has 5-7 well-formed steps", () => {
    expect(TOUR_STEPS.length).toBeGreaterThanOrEqual(5);
    expect(TOUR_STEPS.length).toBeLessThanOrEqual(7);
    for (const s of TOUR_STEPS) {
      expect(s.id).toBeTruthy();
      expect(s.title).toBeTruthy();
      expect(s.body).toBeTruthy();
    }
  });
});
