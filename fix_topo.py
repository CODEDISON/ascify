#!/usr/bin/env python3
"""Fix graphTopologicalOrder (dependency-satisfied selection, not inverted
Kahn indegree) and re-anchor."""
import pathlib, hashlib, subprocess, re

ROOT = pathlib.Path(__file__).resolve().parent
HEADER = ROOT / "include/ascify/ascify_cuda_compat.hpp"

t = HEADER.read_text()
old = """inline aclError graphTopologicalOrder(AscifyGraphObj *graph,
                                      AscifyGraphExecObj *exec) {
  size_t indegree[64] = {};
  bool consumed[64] = {};
  for (size_t i = 0; i < graph->node_count; ++i) {
    const GraphNode &node = graph->nodes[i];
    for (size_t d = 0; d < node.dep_count; ++d) {
      if (node.deps[d] >= graph->node_count)
        return ACL_ERROR_RT_PARAM_INVALID;
      ++indegree[node.deps[d]];
    }
  }
  // Kahn's algorithm preferring lower node indices: deterministic and, for
  // graphs built in submission order, identical to the recorded order.
  size_t produced = 0;
  while (produced < graph->node_count) {
    bool progressed = false;
    for (size_t i = 0; i < graph->node_count; ++i) {
      if (consumed[i] || indegree[i] != 0) continue;
      consumed[i] = true;
      exec->order[produced++] = static_cast<uint32_t>(i);
      progressed = true;
    }
    if (!progressed) return ACL_ERROR_RT_PARAM_INVALID;  // cycle
  }
  exec->order_count = produced;
  return ACL_SUCCESS;
}"""
new = """inline aclError graphTopologicalOrder(AscifyGraphObj *graph,
                                      AscifyGraphExecObj *exec) {
  bool ordered[64] = {};
  // Repeated selection of the lowest-index node whose dependencies are all
  // ordered: deterministic, and for graphs built in submission order
  // identical to the recorded order. A full pass without progress is a
  // dependency cycle.
  size_t produced = 0;
  while (produced < graph->node_count) {
    bool progressed = false;
    for (size_t i = 0; i < graph->node_count; ++i) {
      if (ordered[i]) continue;
      bool ready = true;
      for (size_t d = 0; d < graph->nodes[i].dep_count; ++d) {
        const uint32_t dependency = graph->nodes[i].deps[d];
        if (dependency >= graph->node_count)
          return ACL_ERROR_RT_PARAM_INVALID;
        if (!ordered[dependency]) {
          ready = false;
          break;
        }
      }
      if (!ready) continue;
      ordered[i] = true;
      exec->order[produced++] = static_cast<uint32_t>(i);
      progressed = true;
    }
    if (!progressed) return ACL_ERROR_RT_PARAM_INVALID;  // cycle
  }
  exec->order_count = produced;
  return ACL_SUCCESS;
}"""
assert t.count(old) == 1, f"anchor {t.count(old)}"
t = t.replace(old, new)
HEADER.write_text(t)
print("topo fixed")

subprocess.run(["python3", "/tmp/regen_ids.py"], cwd=ROOT, check=True)
NEW_SIZE = str(HEADER.stat().st_size)
NEW_SHA = hashlib.sha256(HEADER.read_bytes()).hexdigest()

# Discover the CURRENT anchor size dynamically from AscifyAction.cpp.
a = (ROOT / "src/AscifyAction.cpp").read_text()
m = re.search(r"contents\.size\(\) != (\d+)\)", a)
OLD_SIZE = m.group(1)
m2 = re.search(r'sha256Equals\(\s*contents,\s*"([0-9a-f]{64})"', a)
OLD_SHA = m2.group(1)
print("old anchors:", OLD_SIZE, OLD_SHA[:8], "-> new:", NEW_SIZE, NEW_SHA[:8])

for rel in ["src/AscifyAction.cpp",
            "tests/rewrite/check_sample_helper_compat.sh",
            "tests/frontend_compat/check_frontend_host_math.py"]:
    p = ROOT / rel
    s = p.read_text()
    s = s.replace(OLD_SIZE, NEW_SIZE).replace(OLD_SHA, NEW_SHA)
    p.write_text(s)
    print("re-anchored", rel)
