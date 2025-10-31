import argparse
from typing import List, Tuple


HEADER_TEXT = (
    "Here is an example of how to plan your tool call paths in a cost-optimal way for your reference. "
    "You should adapt to the task and available tools instead of memorizing this example."
)


def _letters(n: int) -> List[str]:
    """Return the first n uppercase single-letter tool names: A, B, C, ...

    The benchmark examples only use single letters, so constrain to 26.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if n > 26:
        raise ValueError("This generator supports up to 26 atomic tools (A..Z)")
    return [chr(ord("A") + i) for i in range(n)]


def _join_cost_name(name: str) -> str:
    return f"{name}(Cost: Cost_{name})"


def _contiguous_composites(names: List[str]) -> List[str]:
    """All contiguous composite tool names of length 2..(n-1), left-to-right by length then start.

    Example for [A,B,C,D,E]: AB, BC, CD, DE, ABC, BCD, CDE, ABCD, BCDE
    """
    n = len(names)
    out: List[str] = []
    for length in range(2, max(2, n)):
        if length >= n:
            break
        for start in range(0, n - length + 1):
            out.append("".join(names[start : start + length]))
    return out


def _enumerate_compositions(n: int) -> List[List[int]]:
    """Enumerate all compositions of n into positive integers, excluding single-part [n].

    Returns lists of segment lengths (e.g., [2,1,2] for n=5), where each part >= 1.
    """
    results: List[List[int]] = []

    def dfs(remaining: int, acc: List[int]) -> None:
        if remaining == 0:
            if len(acc) >= 2:  # exclude the single block of length n (no cuts)
                results.append(acc.copy())
            return
        # Choose next part size between 1 and remaining
        for part in range(1, remaining + 1):
            acc.append(part)
            dfs(remaining - part, acc)
            acc.pop()

    dfs(n, [])
    return results


def _first_composite_index(blocks: List[int]) -> int:
    for i, b in enumerate(blocks):
        if b > 1:
            return i
    return len(blocks)


def _sorted_compositions(n: int) -> List[List[int]]:
    """Return compositions sorted in a stable, readable order.

    Strategy:
    - More blocks first (i.e., finer splits earlier)
    - Among ties, earlier composite appears first
    - As final tie-breaker, lexicographic by block sizes
    This matches the spirit of examples though not necessarily their exact order.
    """
    comps = _enumerate_compositions(n)
    return sorted(
        comps,
        key=lambda blocks: (
            -len(blocks),
            _first_composite_index(blocks),
            tuple(blocks),
        ),
    )


def _blocks_to_names(names: List[str], blocks: List[int]) -> Tuple[List[str], List[str]]:
    """Map block sizes into tool names and cost token names.

    Returns (display_parts, cost_parts) where display_parts are like
    ["AB(Cost: Cost_AB)", "C(Cost: Cost_C)"] and cost_parts are like
    ["Cost_AB", "Cost_C"].
    """
    display_parts: List[str] = []
    cost_parts: List[str] = []
    i = 0
    for size in blocks:
        name = "".join(names[i : i + size])
        display_parts.append(_join_cost_name(name))
        cost_parts.append(f"Cost_{name}")
        i += size
    return display_parts, cost_parts


def generate_example_content(num_atoms: int) -> str:
    """Generate the EXAMPLE_CONTENT text for a given number of atomic tools.

    num_atoms >= 3 is expected to mirror EXAMPLE_CONTENT_0 (3 atoms), 1 (4 atoms), etc.
    """
    if num_atoms < 2:
        raise ValueError("num_atoms must be >= 2")

    names = _letters(num_atoms)
    atomic_seq = ", ".join(_join_cost_name(n) for n in names)

    available_tools = names + _contiguous_composites(names)
    available_str = ", ".join(_join_cost_name(t) for t in available_tools)

    # Build paths
    lines: List[str] = []
    for idx, blocks in enumerate(_sorted_compositions(num_atoms), start=1):
        display_parts, cost_parts = _blocks_to_names(names, blocks)
        path_str = (
            f"<path> {idx}. "
            f"{' -> '.join(display_parts)}. "
            f"Total Cost: {' + '.join(cost_parts)}.</path>"
        )
        lines.append(path_str)

    example = (
        "<example>\n"
        f"{HEADER_TEXT}\n\n"
        "If:\n"
        f"1. The basic atomic tool calling sequence is: {atomic_seq}.\n"
        f"2. The available tools are: {available_str}. "
        "Composite tools are those whose names contain at least two letters; each letter represents an atomic tool included within the composite, "
        "while their costs are not necessarily the sum of their component atomic tools (e.g., 'AB' is equivalent in effect to performing A then B, but Cost_AB may differ from Cost_A + Cost_B).\n\n"
        "Then you should list out all possible tool calling paths first:\n"
        + "\n".join(lines)
        + "\n\nAt last, you should select and execute the path with the lowest total cost.\n"
        "</example>"
    )
    return example


def example_for_refinement_level(refinement_level: int) -> str:
    """Return example content by refinement level.

    Mapping follows existing usage where index == refinement_level and
    num_atoms = refinement_level + 3.
    """
    if refinement_level < 0:
        raise ValueError("refinement_level must be >= 0")
    return generate_example_content(refinement_level + 3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate EXAMPLE_CONTENT-style examples.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--index", type=int, help="Example index i, where num_atoms = i + 3 (i>=0)")
    group.add_argument("--refinement-level", type=int, help="Refinement level r, where num_atoms = r + 3 (r>=0)")
    group.add_argument("--num-atoms", type=int, help="Number of atomic tools (e.g., 7 for A..G)")
    parser.add_argument(
        "--assign",
        action="store_true",
        help="Wrap output in a Python assignment like EXAMPLE_CONTENT_i = \"\"\"...\"\"\"",
    )

    args = parser.parse_args()
    if args.index is not None:
        if args.index < 0:
            raise SystemExit("--index must be >= 0")
        num_atoms = args.index + 3
        var_name = f"EXAMPLE_CONTENT_{args.index}"
    elif args.refinement_level is not None:
        if args.refinement_level < 0:
            raise SystemExit("--refinement-level must be >= 0")
        num_atoms = args.refinement_level + 3
        var_name = f"EXAMPLE_CONTENT_{args.refinement_level}"
    else:
        if args.num_atoms is None or args.num_atoms < 2:
            raise SystemExit("--num-atoms must be >= 2")
        num_atoms = args.num_atoms
        var_name = None

    content = generate_example_content(num_atoms)
    if args.assign:
        if var_name is None:
            var_name = f"EXAMPLE_CONTENT_{num_atoms - 3}"
        print(f"{var_name} = \"\"\"{content}\"\"\"")
    else:
        print(content)


if __name__ == "__main__":
    main()


