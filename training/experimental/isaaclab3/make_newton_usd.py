"""Make a copy of chhaya.usd that Isaac Lab 3.0's Newton backend can load (kit-less, no Isaac Sim).

    python make_newton_usd.py ../../envs/assets/robotV2/chhaya.usd chhaya_newton.usda

chhaya.usd is written for PhysX. Newton's USD parser (Newton 1.6) differs in three ways that matter here:
  1. The leg links' colliders carry PhysicsCollisionAPI on an Xform ABOVE the mesh (old Isaac Sim URDF
     importer layout, instanced). PhysX accepts that; Newton skips those prims, so the legs get no collision
     shapes and the robot falls through the ground. -> de-instance, move the API onto the Mesh prims.
  2. `convexDecomposition` is honoured literally (CoACD): ~950 convex pieces per robot. -> convexHull.
  3. The visual materials reference Isaac Sim's built-in OmniPBR.mdl, which does not exist without Isaac
     Sim. -> drop the reference.
Needs only OpenUSD (pxr), e.g. `pip install usd-core`.
"""
import sys

from pxr import Usd, UsdPhysics


def convert(src: str, dst: str) -> dict:
    stage = Usd.Stage.Open(src)
    inst = [p for p in stage.Traverse() if p.IsInstance()]
    for p in inst:
        p.SetInstanceable(False)
    stage.Flatten().Export(dst)
    text = open(dst).read()
    open(dst, "w").write("\n".join(line for line in text.split("\n") if "info:mdl:sourceAsset" not in line))

    out = Usd.Stage.Open(dst)
    moved = hulls = 0
    for prim in list(out.Traverse()):
        if prim.GetTypeName() == "Xform" and prim.HasAPI(UsdPhysics.CollisionAPI):
            for child in Usd.PrimRange(prim):
                if child.GetTypeName() == "Mesh":
                    UsdPhysics.CollisionAPI.Apply(child)
                    UsdPhysics.MeshCollisionAPI.Apply(child).CreateApproximationAttr("convexHull")
                    moved += 1
            prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
            prim.RemoveAPI(UsdPhysics.CollisionAPI)
    for prim in out.Traverse():
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            attr = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr()
            if attr.Get() == "convexDecomposition":
                attr.Set("convexHull")
                hulls += 1
    out.GetRootLayer().Save()
    colliders = sum(1 for p in out.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI))
    return {"deinstanced": len(inst), "collision_api_moved_to_meshes": moved, "decomposition_to_hull": hulls, "colliders": colliders}


if __name__ == "__main__":
    print(convert(sys.argv[1], sys.argv[2]))
