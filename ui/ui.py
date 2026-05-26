import os
import time
import xml.etree.ElementTree as ET
import pygame

# -- display setup ---------------------------------------------
os.environ['SDL_VIDEODRIVER'] = 'x11'
os.environ['DISPLAY']         = ':0'
os.environ["SDL_AUDIODRIVER"] = "dummy"

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
INFERRED_H   = 140   # height of the pinned inferred area


# -------------------------------------------------------------
# InkML export
# -------------------------------------------------------------

def strokes_to_inkml(strokes, canvas_rect):
    
    #Convert a list of strokes to an InkML XML string.
    #Each stroke is a list of (x, y, t) tuples.
    #Coordinates are normalised to the canvas bounding box.
    
    ink = ET.Element('ink', xmlns='http://www.w3.org/2003/InkML')

    definitions = ET.SubElement(ink, 'definitions')
    ctx         = ET.SubElement(definitions, 'context', xml_id='ctx1')
    inkSrc      = ET.SubElement(ctx, 'inkSource', xml_id='inkSrc1')
    channels    = ET.SubElement(inkSrc, 'traceFormat')
    ET.SubElement(channels, 'channel', name='X', type='decimal')
    ET.SubElement(channels, 'channel', name='Y', type='decimal')
    ET.SubElement(channels, 'channel', name='T', type='decimal')

    tg = ET.SubElement(ink, 'traceGroup')

    cx, cy = canvas_rect.x, canvas_rect.y
    cw, ch = canvas_rect.width, canvas_rect.height

    for stroke in strokes:
        if not stroke:
            continue
        trace  = ET.SubElement(tg, 'trace')
        points = []
        for (x, y, t) in stroke:
            nx = round((x - cx) / cw, 6)
            ny = round((y - cy) / ch, 6)
            points.append(f'{nx} {ny} {round(t, 3)}')
        trace.text = ', '.join(points)

    return ET.tostring(ink, encoding='unicode', xml_declaration=True)


# -------------------------------------------------------------
# Dummy pipeline  (replace with real model later)
# -------------------------------------------------------------

def run_recognition(inkml_str):
    """
    Stub recognition stage — returns (inferred_latex, error_str).
    Replace this with the real CTC transformer inference call.
    """
    stroke_count = inkml_str.count('<trace>')
    if stroke_count == 0:
        return None, 'Nothing drawn.'
    return r'x^2 + 5x + 6 = 0', None


def run_solve(inferred_latex):
    """
    Stub computation stage — returns (result_str, error_str).
    Replace this with the real SymPy/NumPy computation call.
    TODO: for matrix results use sympy.pretty(matrix, use_unicode=True)
          and split result by newline, rendering each row separately
          in draw_right_panel().
    """
    if not inferred_latex:
        return None, 'No expression to solve.'
    return 'x = -2,  x = -3', None


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

    def begin(self, x, y):
        self.is_drawing = True
        self.cur_stroke = [(x, y, time.time())]

    def move(self, x, y):
        if self.is_drawing:
            self.cur_stroke.append((x, y, time.time()))

    def end(self):
        if self.is_drawing and self.cur_stroke:
            self.strokes.append(self.cur_stroke)
        self.cur_stroke = []
        self.is_drawing = False

    def clear(self):
        self.strokes    = []
        self.cur_stroke = []
        self.is_drawing = False

    def has_content(self):
        return bool(self.strokes)


# -------------------------------------------------------------
# Helpers for rendering
# -------------------------------------------------------------

def draw_button(screen, font, rect, colour, text, text_col=WHITE):
    pygame.draw.rect(screen, colour, rect, border_radius=6)
    lbl = font.render(text, True, text_col)
    screen.blit(lbl, lbl.get_rect(center=rect.center))


def draw_left_panel(screen, font, font_sm,
                    draw_state, canvas_rect,
                    clear_rect, submit_rect, split):
    H = screen.get_height()
    pygame.draw.rect(screen, PANEL_L, (0, 0, split, H))

    # canvas
    pygame.draw.rect(screen, CANVAS_BG,   canvas_rect, border_radius=6)
    pygame.draw.rect(screen, DIVIDER_COL, canvas_rect, width=1, border_radius=6)

    if not draw_state.has_content():
        hint = font_sm.render('draw here', True, LABEL_COL)
        screen.blit(hint, hint.get_rect(center=canvas_rect.center))

    # completed strokes
    for stroke in draw_state.strokes:
        if len(stroke) < 2:
            x, y, _ = stroke[0]
            pygame.draw.circle(screen, STROKE_COL, (int(x), int(y)), STROKE_W)
        else:
            pts = [(int(p[0]), int(p[1])) for p in stroke]
            pygame.draw.lines(screen, STROKE_COL, False, pts, STROKE_W)

    # stroke in progress
    if draw_state.cur_stroke and len(draw_state.cur_stroke) >= 2:
        pts = [(int(p[0]), int(p[1])) for p in draw_state.cur_stroke]
        pygame.draw.lines(screen, STROKE_COL, False, pts, STROKE_W)

    draw_button(screen, font, clear_rect,  BTN_CLEAR,  'clear')
    draw_button(screen, font, submit_rect, BTN_SUBMIT, 'submit')


def draw_right_panel(screen, font, font_sm,
                     history, scroll_offset, current_inferred,
                     solve_rect, clear_r_rect,
                     right_panel_rect, inferred_rect, content_rect):
    W = screen.get_width()
    pygame.draw.rect(screen, PANEL_R, right_panel_rect)

    x0 = right_panel_rect.x + PAD

    # -- pinned inferred expression area -----------------------
    pygame.draw.rect(screen, CANVAS_BG, inferred_rect)
    pygame.draw.line(screen, DIVIDER_COL,
                     (inferred_rect.x,     inferred_rect.bottom),
                     (inferred_rect.right, inferred_rect.bottom), 1)

    if current_inferred:
        lbl = font_sm.render('INFERRED', True, ACCENT)
        screen.blit(lbl, (x0, inferred_rect.y + 8))
        # TODO: replace static text with editable widget once input
        #       method is decided (USB keyboard, soft keyboard, etc.)
        expr = font.render(current_inferred, True, WHITE)
        screen.blit(expr, (x0 + 4,
                           inferred_rect.y + 8 + lbl.get_height() + 4))
    else:
        hint = font_sm.render('inferred expression appears here', True, LABEL_COL)
        screen.blit(hint, hint.get_rect(
            centerx=inferred_rect.centerx,
            centery=inferred_rect.centery))

    # -- scrollable history ------------------------------------
    old_clip = screen.get_clip()
    screen.set_clip(content_rect)

    y = content_rect.y + PAD - scroll_offset

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

                res_txt = font.render(entry.result or '', True, RESULT_COL)
                screen.blit(res_txt, (x0 + 4, y))
                y += res_txt.get_height() + 12

            pygame.draw.line(screen, ENTRY_LINE,
                             (x0, y), (content_rect.right - PAD, y), 1)
            y += 14

    screen.set_clip(old_clip)

    # -- buttons -----------------------------------------------
    draw_button(screen, font, solve_rect,   BTN_SOLVE,  'solve')
    draw_button(screen, font, clear_r_rect, BTN_CLEAR,  'clear')


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------

def main():
    pygame.init()

    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    W, H   = screen.get_size()
    print(f'Screen resolution: {W}x{H}')

    #SPLIT = W // 2 # for 50/50 split but i want it to be a little uneven
    SPLIT = int(W * 0.62)

    # -- left panel geometry -----------------------------------
    left_btn_w  = (SPLIT - BTN_MARGIN * 3) // 2
    canvas_rect = pygame.Rect(
        BTN_MARGIN, BTN_MARGIN,
        SPLIT - BTN_MARGIN * 2,
        H - BTN_H - BTN_MARGIN * 3
    )
    clear_rect  = pygame.Rect(
        BTN_MARGIN,
        H - BTN_H - BTN_MARGIN,
        left_btn_w, BTN_H
    )
    submit_rect = pygame.Rect(
        BTN_MARGIN * 2 + left_btn_w,
        H - BTN_H - BTN_MARGIN,
        left_btn_w, BTN_H
    )

    # -- right panel geometry ----------------------------------
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
    font    = pygame.font.Font(None, 30)
    font_sm = pygame.font.Font(None, 24)

    # -- state -------------------------------------------------
    draw_state       = DrawState()
    history          = []
    scroll_offset    = 0
    scroll_drag_y    = None
    processing       = False
    current_inferred = None   # inferred expression awaiting SOLVE

    def full_redraw():
        screen.fill(BG)
        # centre divider
        pygame.draw.rect(screen, DIVIDER_COL,
                         (SPLIT, 0, DIVIDER_W, H))
        draw_left_panel(screen, font, font_sm,
                        draw_state, canvas_rect,
                        clear_rect, submit_rect, SPLIT)
        draw_right_panel(screen, font, font_sm,
                         history, scroll_offset, current_inferred,
                         solve_rect, clear_r_rect,
                         right_panel_rect, inferred_rect, content_rect)
        pygame.display.flip()

    full_redraw()

    clock   = pygame.time.Clock()
    running = True

    while running:
        dirty = False

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
                    dirty = True

                elif clear_rect.collidepoint(px, py):
                    draw_state.clear()
                    dirty = True

                elif submit_rect.collidepoint(px, py) and not processing:
                    if draw_state.has_content():
                        processing       = True
                        inkml            = strokes_to_inkml(
                                              draw_state.strokes, canvas_rect)
                        inferred, error  = run_recognition(inkml)
                        current_inferred = inferred if not error else None
                        draw_state.clear()
                        processing       = False
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
                    dirty = True

                if scroll_drag_y is not None:
                    delta         = scroll_drag_y - py
                    scroll_offset = max(0, scroll_offset + delta)
                    scroll_drag_y = py
                    dirty         = True

            elif event.type in (pygame.MOUSEBUTTONUP, pygame.FINGERUP):
                draw_state.end()
                scroll_drag_y = None
                dirty = True

        if dirty:
            full_redraw()

        clock.tick(60)

    pygame.quit()


if __name__ == '__main__':
    main()
