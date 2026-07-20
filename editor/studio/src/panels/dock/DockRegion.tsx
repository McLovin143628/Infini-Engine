import { Fragment, useRef } from "react";
import { cn } from "../../lib/utils";
import { crossSizeProp, groupStackDirection, isHorizontalSide } from "./dockGeometry";
import { useDockLayout } from "./dockLayoutStore";
import { DockGroup } from "./DockGroup";
import { GroupSplitter, RegionSplitter } from "./DockSplitter";
import type { DockSide } from "./dockTypes";

/**
 * One dock region (edge of the workspace). ALWAYS rendered — an empty
 * region collapses to zero size but keeps its place in the tree, so React
 * reconciliation never moves the center container between parents (the
 * native viewport must RESIZE, never remount — Spike A invariant).
 *
 * Left/right regions are sidebars whose `size` is a width; top/bottom are
 * bands whose `size` is a height. Groups stack along the main axis by flex
 * weight, separated by weight splitters; the center-facing edge carries
 * the region splitter.
 */
export function DockRegion({ side }: { side: DockSide }) {
  const region = useDockLayout((s) => s.layout.docks[side]);
  const hydrated = useDockLayout((s) => s.hydrated);
  const stackRef = useRef<HTMLDivElement | null>(null);

  const empty = !hydrated || region.groups.length === 0;
  const horizontal = isHorizontalSide(side);

  return (
    <div
      data-dock-region={side}
      aria-hidden={empty || undefined}
      className={cn(
        "relative shrink-0 overflow-visible bg-(--ink-bg-1)",
        empty && "pointer-events-none",
        !empty && side === "left" && "border-r border-(--ink-border)",
        !empty && side === "right" && "border-l border-(--ink-border)",
        !empty && side === "top" && "border-b border-(--ink-border)",
        !empty && side === "bottom" && "border-t border-(--ink-border)",
      )}
      style={{
        // Placed into the DockWorkspace CSS grid by side (grid-area name ==
        // side). left/right span the full height; top/bottom inset into the
        // centre column. Empty regions collapse to 0.
        gridArea: side,
        [crossSizeProp(side)]: empty ? 0 : region.size,
      }}
    >
      {!empty && (
        <>
          <div
            ref={stackRef}
            className="flex h-full w-full min-w-0 overflow-hidden"
            style={{ flexDirection: groupStackDirection(side) }}
          >
            {region.groups.map((group, i) => (
              <Fragment key={group.id}>
                {i > 0 && <GroupSplitter side={side} index={i - 1} containerRef={stackRef} />}
                <div
                  className={cn(
                    "flex min-h-0 min-w-0 overflow-hidden",
                    horizontal ? "w-full" : "h-full",
                  )}
                  style={{ flexGrow: group.weight, flexBasis: 0 }}
                >
                  <DockGroup side={side} group={group} />
                </div>
              </Fragment>
            ))}
          </div>
          <RegionSplitter side={side} />
        </>
      )}
    </div>
  );
}
