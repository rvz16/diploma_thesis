"""
Constraint-Stress-FC: a small controlled function-calling benchmark designed to
elicit *wrong-valid* calls from small models.

Every task: a tool list with several confusable tools (each with its own
enum/int/bool argument schema) and a clear natural-language request whose gold
tool + gold arguments are known. Because decoding is hard-constrained, every
emitted call is schema-valid; correctness is exact-match against gold, so

    wrong_valid == (not correct).

Difficulty levers (per the proposal's Constraint-Stress-FC): many similar tools,
strict enums vs synonym phrasing, indirect tool selection, distractor tools.
"""

from __future__ import annotations

import itertools
import random


def q(v):
    return f'"{v}"'


def enum(vals):
    return [q(v) for v in vals]


def ints(vals):
    return [str(v) for v in vals]


ROOMS = ["living_room", "bedroom", "kitchen", "office", "bathroom"]
ROOM_PHRASE = {"living_room": "living room", "bedroom": "bedroom",
               "kitchen": "kitchen", "office": "office", "bathroom": "bathroom"}
CITIES = ["paris", "london", "tokyo", "berlin", "madrid"]


def build_dataset(seed: int = 0):
    rng = random.Random(seed)
    ex = []

    def add(family, request, tools_spec, gold_name, gold_args):
        ex.append({
            "id": f"{family}-{len(ex):04d}",
            "family": family,
            "system": ("You are a function-calling assistant. Pick exactly one "
                       "tool from the list and emit a single JSON function call "
                       "matching the request."),
            "user": request,
            "tools_spec": tools_spec,
            "gold": {"name": gold_name, "arguments": gold_args},
        })

    # --- Family 1: smart home (turn_on vs turn_off vs set_brightness) ----- #
    sh_tools = [
        ("turn_on_device", [("room", enum(ROOMS)),
                            ("device", enum(["light", "fan", "tv", "heater"]))]),
        ("turn_off_device", [("room", enum(ROOMS)),
                             ("device", enum(["light", "fan", "tv", "heater"]))]),
        ("set_brightness", [("room", enum(ROOMS)),
                            ("level", enum(["low", "medium", "high"]))]),
        ("set_temperature", [("room", enum(ROOMS)),
                             ("degrees", ints([18, 20, 22, 24]))]),
    ]
    sh_desc = ("Tools:\n"
               "- turn_on_device(room, device[light,fan,tv,heater])\n"
               "- turn_off_device(room, device[light,fan,tv,heater])\n"
               "- set_brightness(room, level[low,medium,high])\n"
               "- set_temperature(room, degrees[18,20,22,24])\n")
    sh_reqs = [
        ("Switch on the {dev} in the {rp}.", "turn_on_device",
         lambda r, d, **k: {"room": r, "device": d}),
        ("Please power off the {rp} {dev}.", "turn_off_device",
         lambda r, d, **k: {"room": r, "device": d}),
        ("Make the {rp} lights {lv}.", "set_brightness",
         lambda r, lv, **k: {"room": r, "level": lv}),
        ("Set the {rp} to {deg} degrees.", "set_temperature",
         lambda r, deg, **k: {"room": r, "degrees": str(deg)}),
        ("Dim the {rp} lights to a {lv} setting.", "set_brightness",
         lambda r, lv, **k: {"room": r, "level": lv}),
        ("Turn the {dev} off in the {rp}.", "turn_off_device",
         lambda r, d, **k: {"room": r, "device": d}),
    ]
    for room, (tmpl, name, garg) in itertools.product(ROOMS, sh_reqs):
        dev = rng.choice(["light", "fan", "tv", "heater"])
        lv = rng.choice(["low", "medium", "high"])
        deg = rng.choice([18, 20, 22, 24])
        req = tmpl.format(rp=ROOM_PHRASE[room], dev=dev, lv=lv, deg=deg)
        add("smart_home", sh_desc + "\nRequest: " + req, sh_tools, name,
            garg(r=room, d=dev, lv=lv, deg=deg))

    # --- Family 2: media control (play/pause/volume/skip) ---------------- #
    media_tools = [
        ("play_music", [("genre", enum(["jazz", "rock", "pop", "classical"])),
                        ("volume", enum(["low", "medium", "high"]))]),
        ("set_volume", [("level", enum(["low", "medium", "high"]))]),
        ("skip_track", [("direction", enum(["next", "previous"]))]),
        ("pause_music", []),
    ]
    media_desc = ("Tools:\n"
                  "- play_music(genre[jazz,rock,pop,classical], volume[low,medium,high])\n"
                  "- set_volume(level[low,medium,high])\n"
                  "- skip_track(direction[next,previous])\n"
                  "- pause_music()\n")
    media_reqs = [
        ("Put on some {g} music at {v} volume.", "play_music",
         lambda g, v, **k: {"genre": g, "volume": v}),
        ("Turn the volume {dir}.", "set_volume", None),  # special
        ("Skip to the {nx} song.", "skip_track",
         lambda nx, **k: {"direction": "next" if nx == "next" else "previous"}),
        ("Go back to the previous track.", "skip_track",
         lambda **k: {"direction": "previous"}),
        ("Pause the music for a moment.", "pause_music", lambda **k: {}),
        ("Start playing {g}, keep it {v}.", "play_music",
         lambda g, v, **k: {"genre": g, "volume": v}),
    ]
    for g, v in itertools.product(["jazz", "rock", "pop", "classical"],
                                  ["low", "medium", "high"]):
        for tmpl, name, garg in media_reqs:
            if name == "set_volume":
                lvl = rng.choice(["low", "medium", "high"])
                phrase = {"low": "down", "high": "up", "medium": "to medium"}[lvl]
                req = "Turn the volume " + phrase + "."
                add("media", media_desc + "\nRequest: " + req, media_tools,
                    "set_volume", {"level": lvl})
            else:
                nx = rng.choice(["next", "previous"])
                req = tmpl.format(g=g, v=v, nx=nx)
                add("media", media_desc + "\nRequest: " + req, media_tools,
                    name, garg(g=g, v=v, nx=nx))

    # --- Family 3: travel booking (flight/hotel/train confusion) --------- #
    travel_tools = [
        ("book_flight", [("city", enum(CITIES)),
                         ("seat", enum(["economy", "business"]))]),
        ("book_train", [("city", enum(CITIES)),
                        ("seat", enum(["economy", "business"]))]),
        ("book_hotel", [("city", enum(CITIES)),
                        ("room", enum(["single", "double", "suite"]))]),
    ]
    travel_desc = ("Tools:\n"
                   "- book_flight(city, seat[economy,business])\n"
                   "- book_train(city, seat[economy,business])\n"
                   "- book_hotel(city, room[single,double,suite])\n")
    travel_reqs = [
        ("Book me a flight to {cp} in {s} class.", "book_flight",
         lambda c, s, **k: {"city": c, "seat": s}),
        ("Reserve a {s}-class train ticket to {cp}.", "book_train",
         lambda c, s, **k: {"city": c, "seat": s}),
        ("I need a {rm} hotel room in {cp}.", "book_hotel",
         lambda c, rm, **k: {"city": c, "room": rm}),
        ("Get me a rail ticket to {cp}, {s} please.", "book_train",
         lambda c, s, **k: {"city": c, "seat": s}),
    ]
    for c in CITIES:
        for tmpl, name, garg in travel_reqs:
            s = rng.choice(["economy", "business"])
            rm = rng.choice(["single", "double", "suite"])
            cp = c.capitalize()
            req = tmpl.format(cp=cp, s=s, rm=rm)
            add("travel", travel_desc + "\nRequest: " + req, travel_tools,
                name, garg(c=c, s=s, rm=rm))

    # --- Family 4: calendar (create/move/delete) ------------------------- #
    DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday"]
    TIMES = ["morning", "afternoon", "evening"]
    cal_tools = [
        ("create_event", [("day", enum(DAYS)), ("time", enum(TIMES))]),
        ("move_event", [("day", enum(DAYS)), ("time", enum(TIMES))]),
        ("delete_event", [("day", enum(DAYS))]),
    ]
    cal_desc = ("Tools:\n"
                "- create_event(day, time[morning,afternoon,evening])\n"
                "- move_event(day, time[morning,afternoon,evening])\n"
                "- delete_event(day)\n")
    cal_reqs = [
        ("Schedule a meeting on {d} {t}.", "create_event",
         lambda d, t, **k: {"day": d, "time": t}),
        ("Reschedule my {d} meeting to the {t}.", "move_event",
         lambda d, t, **k: {"day": d, "time": t}),
        ("Cancel my appointment on {d}.", "delete_event",
         lambda d, **k: {"day": d}),
        ("Set up a new event for {d} in the {t}.", "create_event",
         lambda d, t, **k: {"day": d, "time": t}),
    ]
    for d in DAYS:
        for tmpl, name, garg in cal_reqs:
            t = rng.choice(TIMES)
            req = tmpl.format(d=d, t=t)
            add("calendar", cal_desc + "\nRequest: " + req, cal_tools, name,
                garg(d=d, t=t))

    # --- Family 5: messaging (email/sms/reminder) ------------------------ #
    PEOPLE = ["alex", "sam", "jordan", "taylor"]
    msg_tools = [
        ("send_email", [("recipient", enum(PEOPLE)),
                        ("priority", enum(["low", "normal", "high"]))]),
        ("send_sms", [("recipient", enum(PEOPLE))]),
        ("create_reminder", [("day", enum(["today", "tomorrow"])),
                             ("priority", enum(["low", "normal", "high"]))]),
    ]
    msg_desc = ("Tools:\n"
                "- send_email(recipient, priority[low,normal,high])\n"
                "- send_sms(recipient)\n"
                "- create_reminder(day[today,tomorrow], priority[low,normal,high])\n")
    msg_reqs = [
        ("Email {p} with {pr} priority.", "send_email",
         lambda p, pr, **k: {"recipient": p, "priority": pr}),
        ("Text {p} a quick message.", "send_sms",
         lambda p, **k: {"recipient": p}),
        ("Remind me {dy} about the {pr}-priority task.", "create_reminder",
         lambda dy, pr, **k: {"day": dy, "priority": pr}),
        ("Send {p} an urgent email.", "send_email",
         lambda p, **k: {"recipient": p, "priority": "high"}),
    ]
    for p in PEOPLE:
        for tmpl, name, garg in msg_reqs:
            pr = rng.choice(["low", "normal", "high"])
            dy = rng.choice(["today", "tomorrow"])
            req = tmpl.format(p=p, pr=pr, dy=dy)
            add("messaging", msg_desc + "\nRequest: " + req, msg_tools, name,
                garg(p=p, pr=pr, dy=dy))

    # dedup by request text (templates can collide on sampled values)
    seen, uniq = set(), []
    for e in ex:
        if e["user"] in seen:
            continue
        seen.add(e["user"])
        uniq.append(e)
    return uniq


if __name__ == "__main__":
    ds = build_dataset()
    from collections import Counter
    print("total:", len(ds))
    print("by family:", Counter(e["family"] for e in ds))
    for e in ds[:3]:
        print("---")
        print(e["user"].splitlines()[-1])
        print("gold:", e["gold"])
