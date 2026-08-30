// COW v1 — "creamer" box-cow (agreed-upon current version, saved 2026-08-30)
// Usage: drop `buildCow()` into the page, add the meshes to a THREE.Group,
// position at `posOf(az, lat)`, orient via orientCow(az, lat) (east/pole basis),
// idle-bob by scaling the surface offset 1+0.02*sin(t) in the render loop.
// NOTE: built at FULL scale (body .34w x .24h x .44d, ~0.48 tall) — the live page
// scales the group with cow.scale.setScalar(0.32) so the cow sits small on the
// unit sphere. If this version scales differently, adjust that factor.
function buildCow(){
  const g = new THREE.Group();
  const cream = new THREE.MeshStandardMaterial({color:0xf2e6cf});
  const brown = new THREE.MeshStandardMaterial({color:0x7a4a2b});
  const dark  = new THREE.MeshStandardMaterial({color:0x17120c});
  const body  = new THREE.Mesh(new THREE.BoxGeometry(.34,.24,.44), cream); body.position.set(0,.26,0);
  const neck  = new THREE.Mesh(new THREE.BoxGeometry(.16,.16,.18), cream); neck.position.set(.2,.3,0);
  const head  = new THREE.Mesh(new THREE.BoxGeometry(.18,.17,.2),  cream); head.position.set(.35,.36,0);
  const snout = new THREE.Mesh(new THREE.BoxGeometry(.05,.1,.13),  brown); snout.position.set(.445,.33,0);
  const eyeL  = new THREE.Mesh(new THREE.SphereGeometry(.023,10,8), dark);
  eyeL.position.set(.35,.37,.09);
  const eyeR  = eyeL.clone(); eyeR.position.z = -.09;
  const earL  = new THREE.Mesh(new THREE.BoxGeometry(.09,.035,.045), cream);
  earL.position.set(.3,.45,.105); earL.rotation.z = .35;
  const earR  = new THREE.Mesh(new THREE.BoxGeometry(.09,.035,.045), cream);
  earR.position.set(.3,.45,-.105); earR.rotation.z = -.35;
  const hornL = new THREE.Mesh(new THREE.ConeGeometry(.026,.055,8), brown);
  hornL.position.set(.35,.46,.07); hornL.rotation.z = .45;
  const hornR = new THREE.Mesh(new THREE.ConeGeometry(.026,.055,8), brown);
  hornR.position.set(.35,.46,-.07); hornR.rotation.z = -.45;
  const legMat = cream;
  for (const [x,z] of [[-.09,-.13],[.07,-.13],[-.09,.13],[.07,.13]]){
    const leg = new THREE.Mesh(new THREE.CylinderGeometry(.028,.034,.2,8), legMat);
    leg.position.set(x,.1,z); g.add(leg);
  }
  const tail = new THREE.Mesh(new THREE.CylinderGeometry(.01,.016,.2,6), brown);
  tail.position.set(-.2,.36,0); tail.rotation.x = 1.1; g.add(tail);
  for (const m of [body,neck,head,snout,eyeL,eyeR,earL,earR,hornL,hornR]) g.add(m);
  return g;
}