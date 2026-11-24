import type { Entity } from "./types";

export interface TreeNode {
  entity: Entity;
  children: TreeNode[];
}

export interface PlacedNode {
  id: string;
  x: number; // generation column
  y: number; // row (fractional for parents: centred over their children)
  entity: Entity;
}

export interface Layout {
  nodes: PlacedNode[];
  links: { from: PlacedNode; to: PlacedNode }[];
  width: number; // generations
  height: number; // rows
}

/** Parent -> children forest. Generation-0 entities (and any orphan whose parent is missing) are roots. */
export function buildForest(entities: Entity[]): TreeNode[] {
  const byId = new Map<string, TreeNode>();
  for (const e of entities) byId.set(e.id, { entity: e, children: [] });
  const roots: TreeNode[] = [];
  for (const e of entities) {
    const node = byId.get(e.id)!;
    const parent = e.parent_id ? byId.get(e.parent_id) : undefined;
    if (parent) parent.children.push(node);
    else roots.push(node);
  }
  const sort = (ns: TreeNode[]) => {
    ns.sort((a, b) => a.entity.id.localeCompare(b.entity.id));
    ns.forEach((n) => sort(n.children));
  };
  sort(roots);
  return roots;
}

/** Tidy layout: leaves get consecutive rows in depth-first order; parents sit at the mean row of their children. */
export function layoutForest(forest: TreeNode[]): Layout {
  const nodes: PlacedNode[] = [];
  const links: Layout["links"] = [];
  let nextRow = 0;
  let maxGen = 0;
  const place = (n: TreeNode): PlacedNode => {
    const kids = n.children.map(place);
    const y = kids.length ? kids.reduce((s, k) => s + k.y, 0) / kids.length : nextRow++;
    const placed: PlacedNode = { id: n.entity.id, x: n.entity.generation, y, entity: n.entity };
    maxGen = Math.max(maxGen, n.entity.generation);
    nodes.push(placed);
    for (const k of kids) links.push({ from: placed, to: k });
    return placed;
  };
  forest.forEach(place);
  return { nodes, links, width: maxGen + 1, height: Math.max(nextRow, 1) };
}
