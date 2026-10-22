"""Look and feel shared by the Streamlit app: the calm page style and the 'whiteboard' lesson card."""

PAGE_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Kalam:wght@400;700&display=swap');
  .block-container { max-width: 780px; padding-top: 2.5rem; }
  .stMarkdown p, .stMarkdown li { font-size: 1.06rem; line-height: 1.75; }
  h3 { margin-top: 1.4rem !important; color: #2f5d50; }
  .tutor-note { color: #5b6b66; font-size: 0.95rem; }

  /* The lesson, laid out like a worked solution on a whiteboard. */
  .board {
    background: #ffffff;
    border: 1px solid #e4e1d8;
    border-radius: 16px;
    padding: 1.3rem 1.7rem 1.1rem;
    box-shadow: 0 2px 10px rgba(40, 60, 50, 0.06);
  }
  .board.handwritten h3,
  .board.handwritten strong { font-family: 'Kalam', 'Comic Sans MS', cursive; font-weight: 700; }
  .board.handwritten h3 { font-size: 1.55rem; color: #23453b; border-bottom: 2px dashed #d9e6e0; padding-bottom: .2rem; }
  .board.handwritten strong { font-size: 1.12rem; color: #2f5d50; }
  .board .katex-display { margin: 0.5rem 0 0.9rem; overflow-x: auto; overflow-y: hidden; padding: .2rem 0; }
  .board .katex { font-size: 1.18em; color: #1f2a27; }
  .board .katex .boxed { border-color: #3f7d6a; border-width: 2px; padding: .25em .5em; }
</style>
"""


def lesson_html(markdown: str, handwritten: bool = True) -> str:
    """Wrap a lesson's Markdown in the whiteboard card. Blank lines keep the Markdown (and math) rendering."""
    return f'<div class="board{" handwritten" if handwritten else ""}">\n\n{markdown}\n\n</div>'
