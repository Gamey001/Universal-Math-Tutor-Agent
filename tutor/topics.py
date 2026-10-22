"""Known math topics, with ready-made proxy problems.

When a photo is too blurry to read the exact numbers, the tutor teaches the
same idea with a clean "proxy" problem. Proxies for known topics come from
here (fast and predictable); every proxy is still verified by the math tool.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Proxy:
    problem_text: str
    expression: str


@dataclass(frozen=True)
class Topic:
    key: str
    name: str
    keywords: tuple[str, ...]
    proxies: tuple[Proxy, ...]


TOPICS: tuple[Topic, ...] = (
    Topic("fractions", "Fractions", ("fraction", "numerator", "denominator", "half", "quarter"), (
        Proxy("You eat 1/4 of a pizza, and your friend eats 1/8. How much of the pizza is gone?", "1/4 + 1/8"),
        Proxy("A jug is 2/3 full. You pour out 1/6 of the jug. How full is it now?", "2/3 - 1/6"),
    )),
    Topic("percentages", "Percentages", ("percent", "%", "discount", "interest"), (
        Proxy("A $40 shirt is 25% off. How many dollars do you save?", "25% of 40"),
        Proxy("You answered 18 out of 20 questions correctly. What percent is that?", "18/20 * 100"),
    )),
    Topic("decimals", "Decimals", ("decimal", "point", "tenth", "hundredth"), (
        Proxy("An apple costs $0.75 and a banana costs $0.40. How much for both?", "0.75 + 0.40"),
    )),
    Topic("differential_equations", "Differential equations",
          ("differential equation", "dy/dx", "ode", "y'", "growth and decay"), (
        Proxy("A population grows at a rate equal to twice its size, starting at 3. Find y(x).", "y' = 2y, y(0) = 3"),
        Proxy("Solve y'' + y = 0 with y(0) = 0 and y'(0) = 1.", "y'' + y = 0, y(0) = 0, y'(0) = 1"),
    )),
    Topic("derivatives", "Derivatives", ("derivative", "differentiat", "d/dx", "slope of a curve", "rate of change",
                                         "chain rule", "product rule", "quotient rule", "calculus"), (
        Proxy("How fast is x^3 changing at any point? Find its derivative.", "derivative of x^3"),
        Proxy("Find the derivative of x^2 sin(x).", "derivative of x^2 sin(x)"),
        Proxy("Find the derivative of sin(x^2).", "derivative of sin(x^2)"),
    )),
    Topic("integrals", "Integrals", ("integra", "antiderivative", "area under", "∫"), (
        Proxy("Find the area under y = x^2 between x = 0 and x = 3.", "integral from 0 to 3 of x^2 dx"),
        Proxy("Integrate 2x cos(x^2).", "integral of 2x cos(x^2) dx"),
        Proxy("Integrate x e^x.", "integral of x e^x dx"),
    )),
    Topic("limits", "Limits", ("limit", "approaches", "tends to", "l'hopital", "l'hôpital"), (
        Proxy("What value does sin(x)/x get closer to as x gets close to 0?", "limit of sin(x)/x as x -> 0"),
        Proxy("What happens to (x^2 - 1)/(x - 1) as x gets close to 1?", "limit of (x^2 - 1)/(x - 1) as x -> 1"),
    )),
    Topic("complex_numbers", "Complex numbers", ("complex", "imaginary", "argand", "conjugate", "modulus"), (
        Proxy("Add the complex numbers 2 + 3i and 1 - i.", "(2 + 3i) + (1 - i)"),
        Proxy("Multiply 1 + 2i by 3 - i.", "(1 + 2i)(3 - i)"),
        Proxy("How far is 3 + 4i from zero (its modulus)?", "abs(3 + 4i)"),
    )),
    Topic("matrices", "Matrices", ("matri", "determinant", "eigen", "inverse", "transpose", "linear algebra"), (
        Proxy("Find the determinant of the matrix [[2, 1], [5, 3]].", "det [[2, 1], [5, 3]]"),
        Proxy("Multiply [[1, 2], [3, 4]] by [[2, 0], [1, 2]].", "[[1, 2], [3, 4]] * [[2, 0], [1, 2]]"),
        Proxy("Find the inverse of [[4, 7], [2, 6]].", "inverse [[4, 7], [2, 6]]"),
    )),
    Topic("simultaneous_equations", "Simultaneous equations",
          ("simultaneous", "system of equations", "systems of equations", "two equations", "elimination"), (
        Proxy("Two numbers add up to 10, and one is 2 more than the other. What are they?",
              "x + y = 10; x - y = 2"),
        Proxy("Pens cost $2 and notebooks $3. You buy 7 things for $17. How many of each?",
              "p + n = 7; 2p + 3n = 17"),
    )),
    Topic("quadratics", "Quadratic equations", ("quadratic", "squared", "parabola", "factoring", "factorising"), (
        Proxy("A square garden has an area of 49 square meters. How long is each side?", "x^2 = 49"),
        Proxy("Find the numbers x that make x² − 5x + 6 equal to zero.", "x^2 - 5x + 6 = 0"),
    )),
    Topic("linear_equations", "Solving for an unknown (linear equations)",
          ("linear", "equation", "solve for", "solving for", "for x", "unknown", "variable", "algebra"), (
        Proxy("A mystery bag of marbles, doubled, plus 3 loose marbles, makes 11. How many marbles are in the bag?",
              "2x + 3 = 11"),
        Proxy("Three equal boxes of crayons and 4 extra crayons make 19 crayons. How many are in each box?",
              "3x + 4 = 19"),
    )),
    Topic("inequalities", "Inequalities", ("inequalit", "greater than", "less than"), (
        Proxy("You have $20 and each comic costs $4. You also want to keep at least $4. "
              "How many comics (x) could you buy?", "4x + 4 <= 20"),
    )),
    Topic("exponents", "Exponents and powers", ("exponent", "power", "squared", "cube", "indices"), (
        Proxy("A bacteria colony doubles every hour. Starting from 1, how many are there after 5 hours?", "2^5"),
    )),
    Topic("order_of_operations", "Order of operations", ("order of operations", "bodmas", "pemdas", "brackets"), (
        Proxy("You buy 3 packs of 4 stickers and get 2 bonus stickers. How many stickers?", "3 * 4 + 2"),
    )),
    Topic("multiplication_division", "Multiplication and division",
          ("multipl", "times", "divi", "product", "quotient", "share"), (
        Proxy("24 cookies are shared equally among 6 friends. How many does each friend get?", "24 / 6"),
        Proxy("There are 7 rows of chairs with 8 chairs in each row. How many chairs?", "7 * 8"),
    )),
    Topic("addition_subtraction", "Addition and subtraction",
          ("addition", "subtract", "sum", "difference", "plus", "minus", "arithmetic"), (
        Proxy("You have 15 stickers and get 27 more. How many stickers now?", "15 + 27"),
    )),
    Topic("ratios", "Ratios and proportions", ("ratio", "proportion", "rate", "scale"), (
        Proxy("A recipe uses 2 cups of flour for every 3 eggs. How much flour for 9 eggs?", "2/3 * 9"),
    )),
    Topic("factorials_counting", "Factorials and counting",
          ("factorial", "permutation", "combination", "arrange", "choose", "counting"), (
        Proxy("In how many different orders can 4 friends stand in a line for a photo?", "4!"),
        Proxy("A pizza shop has 5 toppings. How many ways can you pick 2 of them?", "5C2"),
    )),
    Topic("angles_triangles", "Angles and triangles",
          ("angle", "triangle", "degree", "geometry", "parallel", "vertical", "polygon"), (
        Proxy("A triangle has angles of 50° and 60°. What is the third angle?", "50 + 60 + x = 180"),
        Proxy("Two straight roads cross. One of the angles where they meet is 70°. "
              "What is the angle right next to it?", "70 + x = 180"),
    )),
    Topic("area_perimeter", "Area and perimeter", ("area", "perimeter", "rectangle", "square"), (
        Proxy("A rug is 6 m long and 4 m wide. How much floor does it cover (its area)?", "6 * 4"),
        Proxy("A fence goes all the way around a 6 m by 4 m garden. How long is the fence?", "2 * (6 + 4)"),
    )),
)

# Order matters in match_topic: specific topics (quadratics) come before
# broader ones (linear equations) that share keywords like "equation".

# Shown as quick-pick buttons when the tutor asks the learner for context, gentlest first.
TOPIC_CHOICES: tuple[str, ...] = tuple(
    next(t.name for t in TOPICS if t.key == key)
    for key in ("addition_subtraction", "multiplication_division", "fractions", "decimals", "percentages",
                "ratios", "order_of_operations", "exponents", "factorials_counting", "angles_triangles", "area_perimeter", "linear_equations",
                "inequalities", "simultaneous_equations", "quadratics", "complex_numbers", "matrices",
                "limits", "derivatives", "integrals", "differential_equations")
)


def match_topic(raw: str | None) -> Topic | None:
    """Map free text from a model or a learner ("solving for x") onto a known topic."""
    if not raw:
        return None
    text = raw.lower()
    for topic in TOPICS:
        if topic.name.lower() == text or topic.key == text:
            return topic
    for topic in TOPICS:
        if any(word in text for word in topic.keywords):
            return topic
    return None


def pick_proxy(topic: Topic, rng: random.Random | None = None) -> Proxy:
    return (rng or random).choice(topic.proxies)
