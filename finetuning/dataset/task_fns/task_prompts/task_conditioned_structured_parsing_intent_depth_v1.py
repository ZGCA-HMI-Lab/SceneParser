import random


JSON_ONLY_PREFIX = "Return a single JSON object only."


OBJECT_CENTRIC_OBJECT_ONLY_PROMPTS = [
    "Parse [OBJ] in this image and return object bbox only.",
    "Find [OBJ] and output only its object bbox.",
    "Locate [OBJ]. Keep output at object level (object bbox only).",
    "Identify [OBJ] and provide object bbox only.",
    "Detect [OBJ] with object-level output only.",
    "For [OBJ], return object bbox and stop at object level.",
    "Return object-only parse for [OBJ] with object bbox.",
    "Output only the object bbox for [OBJ].",
    "Provide object-level localization for [OBJ] only.",
    "Parse [OBJ] at object granularity and return object bbox.",
    "Find target [OBJ] and return object bbox without deeper expansion.",
    "Locate target [OBJ] and keep result object-only.",
    "Mark [OBJ] with its object box only.",
    "Point out [OBJ] and output just one object-level bbox.",
    "Give me the object box for [OBJ], no part or affordance details.",
    "Return only object coordinates for [OBJ].",
    "Where is [OBJ]? Provide object bbox only.",
    "Show [OBJ] location at object level only.",
    "Identify [OBJ] and avoid fine-grained expansion.",
    "Detect [OBJ] and keep parsing depth minimal (object only).",
    "For [OBJ], output only top-level instance box.",
    "Return object detection style output for [OBJ] (bbox only).",
    "Localize [OBJ] as an object without part or action output.",
    "Find [OBJ] and stop parsing after object bbox.",
    "Object-only query: [OBJ], return bbox.",
    "Given [OBJ], provide only object bounding box.",
    "Produce coarse localization for [OBJ] at object level.",
    "Get object-level result for [OBJ] and omit deeper fields.",
    "Extract [OBJ] object bbox and do not expand hierarchy.",
    "Return [OBJ] as object-only structured output.",
]

OBJECT_CENTRIC_FLEX_PROMPTS = [
    "Parse [OBJ] in this image and return object bbox, plus part bboxes and affordance points when available.",
    "Parse [OBJ]. Always return object bbox; include part and affordance only when reliable.",
    "Find [OBJ] and return adaptive hierarchical parse based on visible evidence.",
    "Locate [OBJ] with object bbox, and expand to part/affordance when supported.",
    "Identify [OBJ] and return object bbox, optionally parts and affordances if visible.",
    "Detect [OBJ] and output object bbox first, then valid part/affordance details when available.",
    "Return structured parse for [OBJ] with adaptive depth (object -> part/affordance as needed).",
    "For [OBJ], return object bbox and include finer levels only when confident.",
    "Find [OBJ]; provide object bbox and conditionally include part boxes and affordance points.",
    "Parse [OBJ] with flexible expansion: stop at object if finer cues are weak.",
    "Locate [OBJ]. Add part and affordance outputs only if they are clearly supported.",
    "Return an evidence-driven parse for [OBJ]: object mandatory, finer levels optional.",
    "Parse [OBJ] in this image.",
    "Find [OBJ] and return its structured parse.",
    "Locate [OBJ] and parse its interaction-relevant structure when visible.",
    "Identify [OBJ] and return object-first hierarchical output.",
    "Detect [OBJ] and parse with adaptive detail level.",
    "For [OBJ], return a practical parse with only visually valid details.",
    "Analyze [OBJ] and expand hierarchy only when supported by visual cues.",
    "Return [OBJ] object box and optionally part/affordance annotations if trustworthy.",
    "Find [OBJ] with adaptive granularity from object to interaction level.",
    "Parse [OBJ] with conservative expansion under uncertainty.",
    "Output object-level result for [OBJ], then include finer structure when evidence is clear.",
    "Return [OBJ] parse with dynamic depth based on confidence.",
    "Where is [OBJ]? Include finer details only when they are reliable.",
    "Detect [OBJ] and add part/action info conditionally.",
    "Give object bbox for [OBJ], then attach valid part and affordance outputs if present.",
    "Return [OBJ] hierarchy with optional deeper fields when visible.",
    "Parse [OBJ] for embodied interaction; keep only evidence-backed details.",
    "Find [OBJ] and return object-level mandatory output, finer levels optional.",
    "Locate [OBJ] and adapt parsing depth to scene quality.",
    "Identify [OBJ] and include part/affordance only if cues are strong.",
    "For [OBJ], produce robust parse with adaptive expansion policy.",
    "Object-first parse for [OBJ]; expand to part/action only when justified.",
    "Return stable [OBJ] parse and avoid unsupported fine-grained hallucinations.",
    "Parse [OBJ] with balanced recall and precision on expansion.",
]

OBJECT_CENTRIC_FULL_PROMPTS = [
    "Parse [OBJ] in this image and return full hierarchical structure with object bbox, part bboxes, and affordance points.",
    "Find [OBJ] and return complete object-part-affordance parse.",
    "Locate [OBJ] and output full hierarchy: object box, part boxes, and affordance points.",
    "Identify [OBJ] with complete hierarchical annotation (object, parts, affordances).",
    "Detect [OBJ] and return fully expanded structured parse.",
    "For [OBJ], return complete interaction-oriented hierarchy with object/part/affordance geometry.",
    "Output full parse depth for [OBJ], including all visible parts and affordance points.",
    "Provide complete hierarchical parsing for [OBJ] with object bbox, part bboxes, and affordance points.",
    "Return full structured output for [OBJ] across object, part, and affordance levels.",
    "Parse [OBJ] exhaustively and include all valid hierarchical elements.",
    "Find target [OBJ] and output complete object-part-affordance structure.",
    "Locate target [OBJ] with maximum parse depth and full geometry fields.",
    "Return comprehensive parse for [OBJ] with object box, all parts, and all affordance points.",
    "Perform full-depth hierarchical parsing on [OBJ].",
    "For [OBJ], enumerate complete part and affordance structure with geometry.",
    "Produce dense hierarchical annotation for [OBJ].",
    "Output full interaction graph for [OBJ] including part/action geometry.",
    "Find [OBJ] and include every visible part plus associated affordance points.",
    "Locate [OBJ] and provide complete object-part-affordance details.",
    "Identify [OBJ] with maximal structural coverage.",
    "Detect [OBJ] and return fully detailed hierarchy for downstream manipulation.",
    "Return complete [OBJ] parse suitable for embodied planning.",
    "Generate full structured representation of [OBJ] at all hierarchy levels.",
    "Parse [OBJ] with full expansion and complete geometric fields.",
    "Provide exhaustive [OBJ] object/part/affordance output.",
    "For [OBJ], return all available hierarchical annotations without truncation.",
    "Find [OBJ] and output complete scene-interaction subtree.",
    "Return full parse chain for [OBJ]: object -> parts -> affordances.",
    "Locate [OBJ] and keep deepest valid parse depth.",
    "Output complete multi-level parse for [OBJ] including interaction points.",
]

SCENE_CENTRIC_PROMPTS = [
    "[SCOPE]=all. Parse all interactive objects in this image.",
    "Parse all interactive objects in this image and return structured outputs.",
    "Return whole-scene hierarchical parsing for all interactive objects.",
    "Analyze the full scene and parse every interactive object.",
    "Parse the entire image and return all interactive objects with hierarchical structure.",
    "Find all interactive objects in the scene and output structured object-part-affordance results.",
    "Provide full-scene parsing for every interactive object visible in the image.",
    "Return scene-level hierarchical annotations for all interactive objects.",
    "Perform global parsing: detect and parse all interactive objects in this image.",
    "Output all interactive object instances in the image with their structured hierarchies.",
    "For the whole image, parse all manipulable or interactive objects.",
    "List and parse every interactive object across the scene.",
    "Do scene-wide structured parsing for all interactive objects.",
    "Parse all target objects in this image at scene scope.",
    "Run all-object hierarchical parsing on this image.",
    "Return complete scene-centric parse for all interactive objects.",
    "Find every interactive object and provide structured outputs for each instance.",
    "Produce a scene-centric hierarchy covering all interactive objects.",
    "Parse the full scene with object boxes and available part/affordance details for all objects.",
    "Global query: parse all interactive objects and return their structured annotations.",
    "Parse all relevant objects in this image under [SCOPE]=all.",
    "Analyze this scene and output a structured parse for each interactive object.",
    "Return all interactive objects with object-level outputs and available finer hierarchy.",
    "Detect and parse all interactive object categories present in this image.",
    "Generate full-scene object-part-affordance parsing outputs.",
    "Across the whole image, return structured parsing for every interactive object.",
    "Scene-level request: parse all interactive objects, not just a single target.",
    "Parse all interactive entities in the image and provide hierarchical results.",
    "Whole-scene parsing: include every interactive object instance with structured output.",
    "For this image, perform all-object structured parsing at global scope.",
]

AFFORDANCE_CENTRIC_PROMPTS = [
    "Find objects or parts that support [ACTION].",
    "Given action [ACTION], return valid interaction targets.",
    "Locate targets that afford [ACTION] and return structured outputs.",
    "Search this image for [ACTION]-compatible targets.",
]

PART_CENTRIC_PROMPTS = [
    "Find [PART] and return its parent object plus related affordance points.",
    "Locate [PART] and recover object-part-affordance linkage.",
    "Retrieve [PART] and return parent object with interaction cues.",
    "Find [PART] in the image and parse its parent object context.",
]


TASK_CONDITIONED_STRUCTURED_PARSING_INTENT_DEPTH_V1 = {
    "object_centric": {
        "object_only": OBJECT_CENTRIC_OBJECT_ONLY_PROMPTS,
        "flex": OBJECT_CENTRIC_FLEX_PROMPTS,
        "full": OBJECT_CENTRIC_FULL_PROMPTS,
    },
    "scene_centric": SCENE_CENTRIC_PROMPTS,
    "affordance_centric": AFFORDANCE_CENTRIC_PROMPTS,
    "part_centric": PART_CENTRIC_PROMPTS,
}


TASK_CONDITIONED_STRUCTURED_PARSING_INTENT_DEPTH_V1_SYSTEM_INSTRUCTION = """Return a single structured output for the requested intent.
- Respect query intent and scope.
- For object-centric parsing, adapt depth by requested mode (object_only/flex/full).
- If target is absent, return an empty object list.
- Include only information supported by visual evidence.
- Do not add extra commentary outside the structured output."""


def sample_object_centric_depth_mode(has_part: bool, has_aff: bool):
    if (not has_part) and (not has_aff):
        return random.choices(["object_only", "flex"], weights=[0.8, 0.2], k=1)[0]
    return random.choices(["full", "flex"], weights=[0.7, 0.3], k=1)[0]


def sample_object_centric_prompt(obj_name: str, depth_mode: str, depth_control_prob: float = 0.25):
    if depth_mode == "object_only":
        bank = OBJECT_CENTRIC_OBJECT_ONLY_PROMPTS
    elif depth_mode == "full":
        bank = OBJECT_CENTRIC_FULL_PROMPTS
    else:
        bank = OBJECT_CENTRIC_FLEX_PROMPTS

    prompt = random.choice(bank).replace("[OBJ]", obj_name)
    if random.random() < max(0.0, min(1.0, depth_control_prob)):
        prompt = f"[DEPTH={depth_mode}] " + prompt
    return JSON_ONLY_PREFIX + "\n" + prompt


def sample_scene_centric_prompt():
    return JSON_ONLY_PREFIX + "\n" + random.choice(SCENE_CENTRIC_PROMPTS)
