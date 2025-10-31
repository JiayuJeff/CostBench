QUERY_GENERATION_INSTRUCTION = "You are a helpful assistant for generating queries. " \
"Generate a search query for a {task} task based on detailed user requirements. The user requirements will be comprised of four dimensions (Category requirement, Tier requirement, Style requirement, Feature_package requirement). The query should be written as a long, self-contained user statement that clearly describes the user's needs and intentions. You should follow these rules:\n" \
    "1. The query clearly discribe the user requirements without any possibilities of misunderstanding. For each requirement dimension, you should clearly distinguish the user required one from any other possible candidates in the generated query. Possible candidates are listed below:\n" \
    "Category requirement: {category_candidates}\n" \
    "Tier requirement: {tier_candidates}\n" \
    "Style requirement: {style_candidates}\n" \
    "Feature_package requirement: {features_candidates}\n" \
    "2. You should use human-like language to express the user requirements. That is to say, you shouldn't use the exact word to describe the user requirements. Instead, you should paraphrase and rephrase the requirements to imply the user needs in a natural way. For example: For if the user has a 'luxury' requirement, then you could say something like 'money is totally not a concern, and I want a extravagant experience'. For some special cases (proper nouns), you can use the exact wording.\n" \
    "3. The query should be concise and to the point, avoiding unnecessary details or overly complex sentences.\n" \
    "4. All the information you could use is from the user preferences. The location and time information are just meaningless placeholders.\n" \
    "Please **DO NOT GENERATE ANYTHING OTHER THAN THE QUERY**. Here is a example:\n" \
    "User prompt: Task: Location search. User requirements: 1. I want the Location category to be 'city'. 2. I want the Location tier to be 'lively town'. 3. I want the Location style to be 'culture'. 4. I want the Location features to include 'nightlife_central'.\n" \
    "Generated query: Lately, I’ve been feeling the urge to get out and experience a change of scenery. I think what I really need is to visit a big city, i mean the kind of place where you can truly feel the energy of the crowds and the pulse of urban life, not a quiet little town. During the day, I’d love to immerse myself in a strong artistic atmosphere: wandering through museums, exploring art galleries, or spending an afternoon in a small, unique theater. What I’m looking for is a destination with a deep cultural foundation, somewhere that can spark a lot of inspiration for me. Of course, the excitement shouldn’t end when the sun goes down. At night, I want the city to stay alive with possibilities — from lively bars to clubs where I can dance, places that keep the evening vibrant and fun. In short, I’m searching for a destination that feels exciting and fulfilling from morning to night.\n" \

QUERY_GENERATION_PROMPT = "User prompt: Task: {task} search. User requirements: 1. I want the {task} category to be {category}. 2. I want the {task} tier to be {tier}. 3. I want the {task} style to be {style}. 4. I want the {task} features to include {features}. Please generate a concise and clear query that reflects these requirements.\nGenerated query: "

QUERY_VALIDATION_INSTRUCTION = (
    "You are a helpful assistant for validating commonsense conflicts in user queries.\n"
    "Given a set of user requirements, determine whether there are any commonsense conflicts among them.\n"
    "Apply **STRICT** checking: even minor inconsistencies should be marked as conflicts.\n"
    "Your response must be either **conflict** or **no conflict**, nothing else.\n"
    "\n"
    "Example:\n"
    "User prompt: Task: Location search. User requirements: "
    "1. I want the Location category to be 'city'. "
    "2. I want the Location tier to be 'secluded_nature'. "
    "3. I want the Location style to be 'adventure'. "
    "4. I want the Location features to include 'nightlife_central'.\n"
    "Generated response: **conflict**\n"
)

QUERY_VALIDATION_PROMPT = "Task: {task} search. User requirements: 1. I want the {task} category to be {category}. 2. I want the {task} tier to be {tier}. 3. I want the {task} style to be {style}. 4. I want the {task} features to include {features}. Please generate a concise and clear query that reflects these requirements. Generated response: "

QUERY_PROMPT = "Please generate a travel plan during {TimeInfo}{LocationPreferenceID} for me according to the user requirements. User requirements: {user_requirements}. "

COUNT_COVERAGE_INSTRUCTION = "You are analyzing a planning text from a model's output. The text lists possible tool-calling paths with costs. Count the number of distinct paths explicitly enumerated (e.g., 'Path 1', '1)', etc.). Ignore introductions or selected paths—only count the listed alternatives. You shouldn't judge if the paths are valid or not. Output a single integer to represent the number of distinct paths. If none, use 0. No extra text. "

COUNT_COVERAGE_PROMPT = "Planning text: {model_plan}. Number of distinct paths: "