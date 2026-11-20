# 04 Chest, arms and head

Do this after the legs and torso are assembled ([03](03_legs.md)).

Parts: the 20 printed parts in [`print/README.md`](../../print/README.md), 8 servos, 2 micro
servos and the screws listed under "fasteners" in [`bom/bom.csv`](../../bom/bom.csv).

## Servo IDs

Servo IDs continue after the legs (13 is `l_hip_yaw`): 14–17 for the left arm (shoulder pitch,
shoulder roll, arm yaw, elbow) and 18–21 for the right arm, as in the
[ID map](../../electronics/wiring/README.md#servo-id-map). Set them one at a time, as with the legs.

## Assembly

Order matters, because some screws get covered:

1. Screw each horn pad (shoulder bracket, elbow link) to its horn **before** mounting the next
   servo, because that servo sits right over the horn screws.
2. Screw the yaw servo to the upper-arm fork's bridge **before** putting the fork onto the roll
   servo.
3. Idler caps go on the roll and elbow servos' back faces. The fork's idler ear slides over the
   stub and is held by an M3 screw and washer (snug, not tight: it has to turn).
4. Claw: screw the micro servo into `claw_base` (2 tab screws), centre it, screw the jaw to its horn
   in the closed position, then bolt the module to the wrist plate from below (4x M3).
5. Electronics rack ([`docs/design.md`](../design.md#electronics)): melt the 4 M3 heat-set inserts into
   the bosses under its plate. Set it into the torso's rim tray on the 4 computer bosses and drill the
   torso's top plate through the rack's 4 anchor holes (Ø3.2; the rack is the drill guide). Bolt it
   down: 4x M2.5 into the bosses, 4x M3 through the top plate with nuts underneath (reach them through
   the torso's open sides).
6. Battery on the rack plate, lengthwise, strap through the two slots. Computer on the deck
   (4x M2.5), USB/Ethernet end to the back.
7. Lower the chest over the rack (the deck passes through the back arch) and screw its 4 feet into
   the inserts (M3x8).

## Head and sensors

- The head bolts to the chest saddle with 4x M3 x 10. Any head with the same flange fits
  (hole pattern in [`docs/design.md`](../design.md)).
- Boards and a camera screw to the plate behind the face disc (10 mm grid of M2.5 holes). Wires go
  down the hollow neck into the chest.
- The face disc presses into the hoop last.

## Claws

Each claw is one module on the wrist plate (4x M3 from below), so it can be swapped for another
tool. The micro servos are driven separately from the bus servos (PWM from the computer) and are
scripted; the walking policy doesn't know about them.

## Software

The policy commands all 20 bus servos; the claws are scripted on top. The on-robot runtime that does
this is **not written yet** ([06](06_software.md)).

## Check

Move every arm joint by hand through its range with the power off: nothing should rub. The design
keeps at least 1 mm between moving parts, so rubbing means a part is mis-seated or a print is
oversize.
