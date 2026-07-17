import os
import time
import threading
import xml.etree.ElementTree as ET
import pygame

# -- display setup ---------------------------------------------
os.environ['SDL_VIDEODRIVER'] = 'x11'
os.environ['DISPLAY']         = ':0'
os.environ["SDL_AUDIODRIVER"] = "dummy" # weird warnings happen without this line even though we obviously dont have audio

# -- colours ---------------------------------------------------
BG          = ( 18,  18,  18)
PANEL_L     = ( 12,  12,  12)
PANEL_R     = ( 22,  22,  22)
CANVAS_BG   = (  8,   8,   8)
DIVIDER_COL = ( 70,  70,  70)
LABEL_COL   = ( 90,  90,  90)
ACCENT      = ( 29, 158, 117) # nice turquoise green
WHITE       = (220, 220, 220)
STROKE_COL  = (220, 220, 220)
BTN_CLEAR   = ( 60,  60,  60)
BTN_SUBMIT  = ( 70,  70, 130) # soft cool blue
BTN_SOLVE   = ACCENT
ENTRY_LINE  = ( 45,  45,  45)
RESULT_COL  = (180, 230, 200)
ERROR_COL   = (230, 100, 100)

# -- layout constants ------------------------------------------
BTN_H        = 48
BTN_MARGIN   = 12
STROKE_W     = 3
PAD          = 16
DIVIDER_W    = 3    # thickness of the centre divider line
INFERRED_H   = 140  # height of the pinned inferred area

# -- ADDED: stroke smoothing factor ----------------------------
SMOOTH = 0.7

# -- ADDED: gap interpolation threshold (pixels) ---------------
INTERP_DIST = 2


# -------------------------------------------------------------
# InkML export - for debugging purposes only
# this function is never actually used by main pipeline
# we use the draw stroke data directly
# this can also be used to generate data if needed
# -------------------------------------------------------------
def strokes_to_inkml(strokes, t0=None):
    ink = ET.Element('ink', xmlns='http://www.w3.org/2003/InkML')
    trace_format = ET.SubElement(ink, 'traceFormat')
    ET.SubElement(trace_format, 'channel', name='X', type='decimal')
    ET.SubElement(trace_format, 'channel', name='Y', type='decimal')
    ET.SubElement(trace_format, 'channel', name='T', type='decimal', units='ms')

    for trace_id, stroke in enumerate(strokes):
        if not stroke:
            continue
        trace  = ET.SubElement(ink, 'trace', id=str(trace_id))
        points = []
        for (x, y, t) in stroke:
            t_ms = round((t - t0) * 1000, 1) if t0 is not None else round(t, 1)
            points.append(f'{round(x, 2)} {round(y, 2)} {t_ms}')
        trace.text = ','.join(points)

    return ET.tostring(ink, encoding='unicode', xml_declaration=True)


# -------------------------------------------------------------
# Recognition pipeline 
# -------------------------------------------------------------
from recognition import run_recognition
from solver import run_solve


# -------------------------------------------------------------
# Background recognition worker
# -------------------------------------------------------------
# run_recognition() runs an ONNX forward pass that can take a noticeable
# fraction of a second on the Pi. Running it inline on the pygame event
# thread freezes the whole UI, so we run it in a daemon thread and let the
# main loop poll `result` for completion.
def _recognize_worker(strokes, result):
    inferred, error = run_recognition(strokes)
    result['inferred'] = inferred
    result['error']    = error
    result['done']     = True


def draw_processing_overlay(screen, font, inferred_rect):
    """Show a 'recognizing…' state in the inferred area while the worker runs."""
    pygame.draw.rect(screen, CANVAS_BG, inferred_rect)
    lbl = font.render('recognizing…', True, ACCENT)
    screen.blit(lbl, lbl.get_rect(center=inferred_rect.center))


# -------------------------------------------------------------
# History entry
# -------------------------------------------------------------
class HistoryEntry:
    def __init__(self, inferred, result, error):
        self.inferred = inferred
        self.result   = result
        self.error    = error


# -------------------------------------------------------------
# Drawing state
# -------------------------------------------------------------
class DrawState:
    def __init__(self):
        self.strokes    = []
        self.cur_stroke = []
        self.is_drawing = False
        self.t0         = None

    def begin(self, x, y):
        self.is_drawing = True
        if not self.strokes and not self.cur_stroke:
            self.t0 = time.time()
        self.cur_stroke = [(x, y, time.time())]

    def move(self, x, y):
        if self.is_drawing:
            if self.cur_stroke:
                lx, ly, _ = self.cur_stroke[-1]
                x = lx + (x - lx) * (1.0 - SMOOTH)
                y = ly + (y - ly) * (1.0 - SMOOTH)
            self.cur_stroke.append((x, y, time.time()))

    def end(self):
        if self.is_drawing and self.cur_stroke:
            self.strokes.append(self.cur_stroke)
        self.cur_stroke = []
        self.is_drawing = False

    def undo(self):
        # Remove the most recently completed stroke, if any.
        if self.strokes:
            self.strokes.pop()

    def clear(self):
        self.strokes    = []
        self.cur_stroke = []
        self.is_drawing = False
        self.t0         = None

    def has_content(self):
        return bool(self.strokes)


# -------------------------------------------------------------
# Gap interpolation helper
# -------------------------------------------------------------
def interpolated_pts(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    if dist <= INTERP_DIST:
        return []
    steps = int(dist / INTERP_DIST)
    return [
        (int(x1 + (x2 - x1) * i / steps),
         int(y1 + (y2 - y1) * i / steps))
        for i in range(1, steps)
    ]


# -------------------------------------------------------------
# Draw a stroke onto a surface with gap interpolation
# -------------------------------------------------------------
def draw_stroke_on_surface(surface, stroke, offset_x, offset_y):
    if not stroke:
        return
    if len(stroke) == 1:
        x, y, _ = stroke[0]
        pygame.draw.circle(surface, STROKE_COL,
                           (int(x - offset_x), int(y - offset_y)), STROKE_W)
        return
    pts = [(int(p[0] - offset_x), int(p[1] - offset_y)) for p in stroke]
    expanded = [pts[0]]
    for i in range(1, len(pts)):
        expanded.extend(interpolated_pts(pts[i - 1], pts[i]))
        expanded.append(pts[i])
    pygame.draw.lines(surface, STROKE_COL, False, expanded, STROKE_W)


# -------------------------------------------------------------
# Expand a screen-space stroke with interpolation for live rendering
# -------------------------------------------------------------
def expanded_screen_pts(stroke):
    pts = [(int(p[0]), int(p[1])) for p in stroke]
    expanded = [pts[0]]
    for i in range(1, len(pts)):
        expanded.extend(interpolated_pts(pts[i - 1], pts[i]))
        expanded.append(pts[i])
    return expanded


# -------------------------------------------------------------
# Helpers for rendering
# -------------------------------------------------------------
def draw_button(screen, font, rect, colour, text, text_col=WHITE):
    pygame.draw.rect(screen, colour, rect, border_radius=6)
    lbl = font.render(text, True, text_col)
    screen.blit(lbl, lbl.get_rect(center=rect.center))


def draw_left_panel(screen, font, font_sm,
                    draw_state, canvas_rect, ink_surface,
                    undo_rect, clear_rect, submit_rect, split):
    H = screen.get_height()
    pygame.draw.rect(screen, PANEL_L, (0, 0, split, H))

    pygame.draw.rect(screen, CANVAS_BG,   canvas_rect, border_radius=6)
    pygame.draw.rect(screen, DIVIDER_COL, canvas_rect, width=1, border_radius=6)

    if not draw_state.has_content() and not draw_state.cur_stroke:
        hint = font_sm.render('draw here', True, LABEL_COL)
        screen.blit(hint, hint.get_rect(center=canvas_rect.center))

    screen.blit(ink_surface, canvas_rect.topleft)
    pygame.draw.rect(screen, DIVIDER_COL, canvas_rect, width=1, border_radius=6)

    if draw_state.cur_stroke:
        if len(draw_state.cur_stroke) == 1:
            x, y, _ = draw_state.cur_stroke[0]
            pygame.draw.circle(screen, STROKE_COL, (int(x), int(y)), STROKE_W)
        else:
            pygame.draw.lines(screen, STROKE_COL, False,
                              expanded_screen_pts(draw_state.cur_stroke), STROKE_W)

    draw_button(screen, font, undo_rect,   BTN_CLEAR,  'undo')
    draw_button(screen, font, clear_rect,  BTN_CLEAR,  'clear')
    draw_button(screen, font, submit_rect, BTN_SUBMIT, 'submit')


def draw_right_panel(screen, font, font_sm, font_mono,
                     history, scroll_offset, current_inferred,
                     solve_rect, clear_r_rect,
                     right_panel_rect, inferred_rect, content_rect):
    W = screen.get_width()
    pygame.draw.rect(screen, PANEL_R, right_panel_rect)

    x0 = right_panel_rect.x + PAD

    pygame.draw.rect(screen, CANVAS_BG, inferred_rect)
    pygame.draw.line(screen, DIVIDER_COL,
                     (inferred_rect.x,     inferred_rect.bottom),
                     (inferred_rect.right, inferred_rect.bottom), 1)

    if current_inferred:
        lbl = font_sm.render('INFERRED', True, ACCENT)
        screen.blit(lbl, (x0, inferred_rect.y + 8))
        expr = font.render(current_inferred, True, WHITE)
        screen.blit(expr, (x0 + 4,
                           inferred_rect.y + 8 + lbl.get_height() + 4))
    else:
        hint = font_sm.render('inferred expression appears here', True, LABEL_COL)
        screen.blit(hint, hint.get_rect(
            centerx=inferred_rect.centerx,
            centery=inferred_rect.centery))

    old_clip = screen.get_clip()
    screen.set_clip(content_rect)

    y = content_rect.y + PAD - scroll_offset
    y_start = y   # track for total content-height computation (scroll clamp)

    if not history:
        empty = font_sm.render('solved results appear here', True, LABEL_COL)
        screen.blit(empty, empty.get_rect(
            centerx=content_rect.centerx,
            centery=content_rect.centery))
    else:
        for entry in reversed(history):
            if entry.error:
                err = font_sm.render(entry.error, True, ERROR_COL)
                screen.blit(err, (x0, y))
                y += err.get_height() + 6
            else:
                inf_lbl = font_sm.render('INFERRED', True, ACCENT)
                screen.blit(inf_lbl, (x0, y))
                y += inf_lbl.get_height() + 4

                inf_txt = font.render(entry.inferred or '', True, WHITE)
                screen.blit(inf_txt, (x0 + 4, y))
                y += inf_txt.get_height() + 10

                res_lbl = font_sm.render('RESULT', True, ACCENT)
                screen.blit(res_lbl, (x0, y))
                y += res_lbl.get_height() + 4

                # -- multiline result rendering (sympy.pretty output) --
                for line in (entry.result or '').split('\n'):
                    if line.strip():
                        res_surf = font_mono.render(line, True, RESULT_COL)
                        screen.blit(res_surf, (x0 + 4, y))
                        y += res_surf.get_height() + 2
                    else:
                        y += font_mono.get_linesize() // 2
                y += 10

            pygame.draw.line(screen, ENTRY_LINE,
                             (x0, y), (content_rect.right - PAD, y), 1)
            y += 14

    screen.set_clip(old_clip)

    draw_button(screen, font, solve_rect,   BTN_SOLVE,  'solve')
    draw_button(screen, font, clear_r_rect, BTN_CLEAR,  'clear')

    # Total pixel height of the rendered history (independent of scroll_offset,
    # since both y and y_start carry the same -scroll_offset shift). Used by the
    # main loop to clamp scrolling so it can't run off the end of the content.
    return max(0, (y - y_start) + PAD)


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
def main():
    pygame.init()

    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    W, H   = screen.get_size()
    print(f'Screen resolution: {W}x{H}')

    SPLIT = int(W * 0.62)

    left_btn_w  = (SPLIT - BTN_MARGIN * 4) // 3   # three buttons: undo | clear | submit
    btn_y       = H - BTN_H - BTN_MARGIN
    canvas_rect = pygame.Rect(
        BTN_MARGIN, BTN_MARGIN,
        SPLIT - BTN_MARGIN * 2,
        H - BTN_H - BTN_MARGIN * 3
    )
    undo_rect   = pygame.Rect(
        BTN_MARGIN,
        btn_y, left_btn_w, BTN_H
    )
    clear_rect  = pygame.Rect(
        BTN_MARGIN * 2 + left_btn_w,
        btn_y, left_btn_w, BTN_H
    )
    submit_rect = pygame.Rect(
        BTN_MARGIN * 3 + left_btn_w * 2,
        btn_y, left_btn_w, BTN_H
    )

    right_x          = SPLIT + DIVIDER_W
    right_w          = W - right_x
    right_panel_rect = pygame.Rect(right_x, 0, right_w, H)
    inferred_rect    = pygame.Rect(right_x, 0, right_w, INFERRED_H)
    btn_row_y        = H - BTN_H - BTN_MARGIN
    content_rect     = pygame.Rect(
        right_x, INFERRED_H,
        right_w, btn_row_y - INFERRED_H - BTN_MARGIN
    )
    right_btn_w  = (right_w - BTN_MARGIN * 3) // 2
    solve_rect   = pygame.Rect(
        right_x + BTN_MARGIN, btn_row_y, right_btn_w, BTN_H
    )
    clear_r_rect = pygame.Rect(
        right_x + BTN_MARGIN * 2 + right_btn_w, btn_row_y, right_btn_w, BTN_H
    )

    # -- fonts -------------------
    _SANS = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    _MONO = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
    try:
        font      = pygame.font.Font(_SANS, 22)
        font_sm   = pygame.font.Font(_SANS, 18)
        font_mono = pygame.font.Font(_MONO, 16)
    except FileNotFoundError:
        font      = pygame.font.Font(None, 28)
        font_sm   = pygame.font.Font(None, 22)
        font_mono = pygame.font.Font(None, 24)

    ink_surface = pygame.Surface((canvas_rect.width, canvas_rect.height))
    ink_surface.fill(CANVAS_BG)

    draw_state       = DrawState()
    history          = []
    scroll_offset    = 0
    scroll_drag_y    = None
    processing       = False
    current_inferred = None
    content_height   = 0       # last rendered history height (for scroll clamp)
    recog_thread     = None    # background recognition worker
    recog_result     = {}      # filled by the worker: inferred / error / done

    def rebuild_ink_surface():
        # Undo can't erase a single stroke from the cumulative bitmap, so redraw
        # the surface from the strokes that remain.
        ink_surface.fill(CANVAS_BG)
        for stroke in draw_state.strokes:
            draw_stroke_on_surface(ink_surface, stroke, canvas_rect.x, canvas_rect.y)

    def full_redraw():
        nonlocal content_height
        screen.fill(BG)
        pygame.draw.rect(screen, DIVIDER_COL,
                         (SPLIT, 0, DIVIDER_W, H))
        draw_left_panel(screen, font, font_sm,
                        draw_state, canvas_rect, ink_surface,
                        undo_rect, clear_rect, submit_rect, SPLIT)
        content_height = draw_right_panel(
                         screen, font, font_sm, font_mono,
                         history, scroll_offset, current_inferred,
                         solve_rect, clear_r_rect,
                         right_panel_rect, inferred_rect, content_rect)
        pygame.display.flip()

    def canvas_redraw():
        pygame.draw.rect(screen, CANVAS_BG, canvas_rect)
        screen.blit(ink_surface, canvas_rect.topleft)
        pygame.draw.rect(screen, DIVIDER_COL, canvas_rect, width=1, border_radius=6)
        if draw_state.cur_stroke:
            if len(draw_state.cur_stroke) == 1:
                x, y, _ = draw_state.cur_stroke[0]
                pygame.draw.circle(screen, STROKE_COL, (int(x), int(y)), STROKE_W)
            else:
                pygame.draw.lines(screen, STROKE_COL, False,
                                  expanded_screen_pts(draw_state.cur_stroke), STROKE_W)
        pygame.display.update(canvas_rect)

    full_redraw()

    clock   = pygame.time.Clock()
    running = True

    while running:
        dirty        = False
        canvas_dirty = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

            elif event.type in (pygame.MOUSEBUTTONDOWN, pygame.FINGERDOWN):
                if event.type == pygame.FINGERDOWN:
                    px, py = int(event.x * W), int(event.y * H)
                else:
                    px, py = event.pos

                if canvas_rect.collidepoint(px, py):
                    draw_state.begin(px, py)
                    canvas_dirty = True

                elif undo_rect.collidepoint(px, py) and not processing:
                    draw_state.undo()
                    rebuild_ink_surface()
                    dirty = True

                elif clear_rect.collidepoint(px, py):
                    draw_state.clear()
                    ink_surface.fill(CANVAS_BG)
                    dirty = True

                elif submit_rect.collidepoint(px, py) and not processing:
                    if draw_state.has_content():
                        # Kick recognition off on a background thread so the UI
                        # stays responsive. `processing` now genuinely gates
                        # re-entry because it persists across frames until the
                        # worker finishes (polled below).
                        processing       = True
                        current_inferred = None
                        strokes_copy     = [list(s) for s in draw_state.strokes]
                        draw_state.clear()
                        ink_surface.fill(CANVAS_BG)
                        recog_result.clear()
                        recog_thread = threading.Thread(
                            target=_recognize_worker,
                            args=(strokes_copy, recog_result),
                            daemon=True,
                        )
                        recog_thread.start()
                        dirty            = True

                elif solve_rect.collidepoint(px, py) and not processing:
                    if current_inferred is not None:
                        processing = True
                        result, error = run_solve(current_inferred)
                        history.append(
                            HistoryEntry(current_inferred, result, error))
                        current_inferred = None
                        scroll_offset    = 0
                        processing       = False
                        dirty            = True

                elif clear_r_rect.collidepoint(px, py):
                    history.clear()
                    current_inferred = None
                    scroll_offset    = 0
                    dirty            = True

                elif content_rect.collidepoint(px, py):
                    scroll_drag_y = py

            elif event.type in (pygame.MOUSEMOTION, pygame.FINGERMOTION):
                if event.type == pygame.FINGERMOTION:
                    px, py = int(event.x * W), int(event.y * H)
                else:
                    px, py = event.pos

                if draw_state.is_drawing and canvas_rect.collidepoint(px, py):
                    draw_state.move(px, py)
                    canvas_dirty = True

                if scroll_drag_y is not None:
                    delta         = scroll_drag_y - py
                    max_scroll    = max(0, content_height - content_rect.height)
                    scroll_offset = min(max(0, scroll_offset + delta), max_scroll)
                    scroll_drag_y = py
                    dirty         = True

            elif event.type in (pygame.MOUSEBUTTONUP, pygame.FINGERUP):
                if draw_state.is_drawing and draw_state.cur_stroke:
                    draw_stroke_on_surface(
                        ink_surface, draw_state.cur_stroke,
                        canvas_rect.x, canvas_rect.y)
                draw_state.end()
                scroll_drag_y = None
                dirty = True

        # Poll the background recognition worker for completion.
        if processing and recog_result.get('done'):
            inferred = recog_result.get('inferred')
            error    = recog_result.get('error')
            current_inferred = inferred if not error else None
            # Debug: surface exactly what the model inferred (or why it failed).
            print(f'[ui] Inferred expression: {current_inferred!r}'
                  + (f'  (error: {error})' if error else ''))
            processing = False
            recog_result.clear()
            dirty = True

        if canvas_dirty:
            canvas_redraw()
        elif dirty:
            full_redraw()
            if processing:
                # Overlay drawn once after the redraw; it persists on screen
                # (no further redraws) until the worker finishes.
                draw_processing_overlay(screen, font, inferred_rect)
                pygame.display.flip()

        clock.tick(60)

    pygame.quit()


if __name__ == '__main__':
    main()
