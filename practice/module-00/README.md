# Module 00 Practice

Extra practice sets for Module 00 live here. These files are for drills created after grading shows that a specific concept needs another pass.

Use this workflow:

1. Ask for a focused set, for example:

   ```text
   Can you make me more module 0 shape-trace problems?
   ```

2. The agent creates the next numbered file, such as `practice/module-00/set-001-shapes.md`.
3. Fill in any `Student answer` sections in that practice file. Blank sections are fine; the grader should skip them rather than treating them as wrong.
4. Ask for grading, for example:

   ```text
   Can you grade practice/module-00/set-001-shapes.md?
   ```

5. The agent can record concise notes in the `Agent feedback` sections, leaving your answers untouched.
6. If you still want another pass, ask for a follow-up set based on what you missed.

Agents should keep generated practice focused, leave answer slots blank, grade only submitted answers by default, and avoid putting full solutions in the student-facing practice file. For repeated loops, agents should inspect previous `Student answer`, `Notes / uncertainty`, and `Agent feedback` sections before generating the next set.
