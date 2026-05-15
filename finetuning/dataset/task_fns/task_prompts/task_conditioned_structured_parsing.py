TASK_CONDITIONED_STRUCTURED_PARSING = [
    "Parse [OBJ] in this image and return its hierarchical structure with object bbox, part bboxes, and affordance points.",
    "parse [OBJ] in this image and return its hierarchical structure with object bbox, part bboxes, and affordance points.",
    "Parse the target [OBJ] and return its object bbox, part bboxes, and affordance points.",
    "parse the target [OBJ] and return its object bbox, part bboxes, and affordance points.",
    "Return the full hierarchical parse for [OBJ], including object bbox, part bboxes, and affordance points.",
    "return the full hierarchical parse for [OBJ], including object bbox, part bboxes, and affordance points.",
    "Describe the structure of [OBJ] with object bbox, part bboxes, and affordance points.",
    "describe the structure of [OBJ] with object bbox, part bboxes, and affordance points.",
    "Find [OBJ] and return a structured parse with object bbox, part bboxes, and affordance points.",
    "find [OBJ] and return a structured parse with object bbox, part bboxes, and affordance points.",
    "Locate [OBJ] and output its object bbox, visible part bboxes, and affordance points.",
    "locate [OBJ] and output its object bbox, visible part bboxes, and affordance points.",
    "Detect [OBJ] and return its hierarchical parse with object bbox, part bboxes, and affordance points.",
    "detect [OBJ] and return its hierarchical parse with object bbox, part bboxes, and affordance points.",
    "Identify [OBJ] and return its object bbox, part bboxes, and affordance points.",
    "identify [OBJ] and return its object bbox, part bboxes, and affordance points.",
    "Find [OBJ] in the image and return its full object-part-affordance structure.",
    "find [OBJ] in the image and return its full object-part-affordance structure.",
    "Locate [OBJ] in the image and return a JSON parse with object bbox, part bboxes, and affordance points.",
    "locate [OBJ] in the image and return a JSON parse with object bbox, part bboxes, and affordance points.",
    "Please detect [OBJ] and return its hierarchical structure with object bbox, part bboxes, and affordance points.",
    "please detect [OBJ] and return its hierarchical structure with object bbox, part bboxes, and affordance points.",
    "Please locate [OBJ] and return object bbox, part bboxes, and affordance points.",
    "please locate [OBJ] and return object bbox, part bboxes, and affordance points.",
    "Please identify [OBJ] and output its object bbox, part bboxes, and affordance points.",
    "please identify [OBJ] and output its object bbox, part bboxes, and affordance points.",
    "Can you find [OBJ] and return its hierarchical parse with object bbox, part bboxes, and affordance points?",
    "can you find [OBJ] and return its hierarchical parse with object bbox, part bboxes, and affordance points?",
    "Can you locate [OBJ] and return its full object-part-affordance structure?",
    "can you locate [OBJ] and return its full object-part-affordance structure?",
    "Point to [OBJ] and return its hierarchical parse with object bbox, part bboxes, and affordance points.",
    "point to [OBJ] and return its hierarchical parse with object bbox, part bboxes, and affordance points.",
    "Mark [OBJ] and output its object bbox, part bboxes, and affordance points.",
    "mark [OBJ] and output its object bbox, part bboxes, and affordance points.",
    "Show where [OBJ] is and return its hierarchical structure with object bbox, part bboxes, and affordance points.",
    "show where [OBJ] is and return its hierarchical structure with object bbox, part bboxes, and affordance points.",
    "Indicate [OBJ] in the image and return its object bbox, part bboxes, and affordance points.",
    "indicate [OBJ] in the image and return its object bbox, part bboxes, and affordance points.",
    "For embodied interaction, parse [OBJ] and return its object bbox, relevant part bboxes, and affordance points.",
    "for embodied interaction, parse [OBJ] and return its object bbox, relevant part bboxes, and affordance points.",
    "Return the interaction-oriented parse of [OBJ] with object bbox, part bboxes, and affordance points.",
    "return the interaction-oriented parse of [OBJ] with object bbox, part bboxes, and affordance points.",
    "Given [OBJ], return the object box, part boxes, and affordance points in JSON.",
    "given [OBJ], return the object box, part boxes, and affordance points in JSON.",
    "Output the structured parse of [OBJ] with object bbox, part bboxes, and affordance points.",
    "output the structured parse of [OBJ] with object bbox, part bboxes, and affordance points.",
    "Parse [OBJ]. Use object bbox, part bboxes, and affordance points only.",
    "parse [OBJ]. Use object bbox, part bboxes, and affordance points only.",
]


# This constant is newly introduced for the task-conditioned parsing branch.
# It is not part of the original codepath and is not wired into the
# existing training config unless we explicitly add a new task function/config.
TASK_CONDITIONED_STRUCTURED_PARSING_SYSTEM_INSTRUCTION = """Return a single JSON object.
Use only the queried target and its directly relevant subtree.
If the queried target is not present, return {\"objects\": []}.
For each matched object, include:
- name
- bbox
- parts
- affordances
Each part should include part_name and bbox.
Each affordance should include action and point when available.
Use null for missing boxes or points.
Do not add extra commentary outside the JSON."""
