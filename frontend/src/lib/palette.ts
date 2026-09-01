import { useEffect, useState } from "react";
/**
 * Colour assignment. Every value here was checked with the dataviz skill's
 * validator (scripts/validate_palette.js) in BOTH modes -- do not hand-edit.
 *
 *   methods (categorical, all-pairs)   PASS light / PASS dark
 *   lora rank ramp   (ordinal, 5 steps) PASS light / PASS dark
 *   qlora rank ramp  (ordinal, 5 steps) PASS light / PASS dark
 *
 * Method carries HUE (identity); rank carries LIGHTNESS (an ordered magnitude).
 * Colour is bound to the variant, never to its position in the selection, so
 * changing the selection never repaints the survivors.
 *
 * Light mode puts aqua at 2.74:1 against the surface -- under the relief rule
 * that obliges visible direct labels and a table view. Both ship.
 */
import type { Method } from "./types";

export const RANKS = [4, 8, 16, 32, 64];

const LIGHT = {
  full_ft: "#1baf7a",
  lora: ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"],
  qlora: ["#ff9067", "#eb6834", "#c94908", "#9e3703", "#752600"],
};

// Dark steps are selected for the dark surface, not flipped from light:
// higher rank stays the higher-contrast end in both modes.
const DARK = {
  full_ft: "#199e70",
  lora: ["#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4"],
  qlora: ["#8a2d01", "#b53e01", "#d95926", "#fb7747", "#ffa98b"],
};

export function variantColor(method: Method, rank: number | null, dark = false): string {
  const set = dark ? DARK : LIGHT;
  if (method === "full_ft" || rank === null) return set.full_ft;
  const i = RANKS.indexOf(rank);
  return set[method][i < 0 ? set[method].length - 1 : i];
}

/** The hue for a method, ignoring rank -- for legends and method chips. */
export function methodColor(method: Method, dark = false): string {
  const set = dark ? DARK : LIGHT;
  return method === "full_ft" ? set.full_ft : set[method][2];
}

/** Tracks the OS colour scheme so charts re-render with the correct steps. */
export function useDarkMode(): boolean {
  const query = "(prefers-color-scheme: dark)";
  const [dark, setDark] = useState(
    () => typeof window !== "undefined" && window.matchMedia(query).matches,
  );
  useEffect(() => {
    const mq = window.matchMedia(query);
    const onChange = (e: MediaQueryListEvent) => setDark(e.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);
  return dark;
}
