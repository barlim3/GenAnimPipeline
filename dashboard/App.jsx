/* ─────────────────────────────────────────────────────────
   GenAnim Dashboard — Single-File React Application
   Multi-Agent Text-to-Motion Generative AI Pipeline GUI
   ───────────────────────────────────────────────────────── */

const { useState, useEffect, useRef, useCallback, useMemo } = React;

// ── Lucide-style SVG Icon Components ──────────────────────

function Icon({ children, size = 16, className = "", strokeWidth = 1.8, ...props }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={strokeWidth}
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      {...props}
    >
      {children}
    </svg>
  );
}

function TerminalIcon(p) {
  return <Icon {...p}><polyline points="4 17 10 11 4 5" /><line x1="12" y1="19" x2="20" y2="19" /></Icon>;
}
function SettingsIcon(p) {
  return <Icon {...p}><circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" /></Icon>;
}
function PlayIcon(p) {
  return <Icon {...p}><polygon points="5 3 19 12 5 21 5 3" /></Icon>;
}
function SquareIcon(p) {
  return <Icon {...p}><rect x="3" y="3" width="18" height="18" rx="2" ry="2" /></Icon>;
}
function ActivityIcon(p) {
  return <Icon {...p}><polyline points="22 12 18 12 15 21 9 3 6 12 2 12" /></Icon>;
}
function DatabaseIcon(p) {
  return <Icon {...p}><ellipse cx="12" cy="5" rx="9" ry="3" /><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" /><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" /></Icon>;
}
function BrainCircuitIcon(p) {
  return <Icon {...p}><path d="M12 4.5a2.5 2.5 0 0 0-4.96-.46 2.5 2.5 0 0 0-1.98 3 2.5 2.5 0 0 0 1.32 4.24 3 3 0 0 0 .34 5.58 2.5 2.5 0 0 0 2.96 3.08A2.5 2.5 0 0 0 12 21.5V12" /><path d="M12 4.5a2.5 2.5 0 0 1 4.96-.46 2.5 2.5 0 0 1 1.98 3 2.5 2.5 0 0 1-1.32 4.24 3 3 0 0 1-.34 5.58 2.5 2.5 0 0 1-2.96 3.08A2.5 2.5 0 0 1 12 21.5V12" /><circle cx="12" cy="12" r="2" /><path d="m7 9 1.5 1.5" /><path d="m17 9-1.5 1.5" /><path d="m7 15 1.5-1.5" /><path d="m17 15-1.5-1.5" /></Icon>;
}
function CheckCircle2Icon(p) {
  return <Icon {...p}><circle cx="12" cy="12" r="10" /><path d="m9 12 2 2 4-4" /></Icon>;
}
function MonitorIcon(p) {
  return <Icon {...p}><rect x="2" y="3" width="20" height="14" rx="2" ry="2" /><line x1="8" y1="21" x2="16" y2="21" /><line x1="12" y1="17" x2="12" y2="21" /></Icon>;
}
function LayersIcon(p) {
  return <Icon {...p}><polygon points="12 2 2 7 12 12 22 7 12 2" /><polyline points="2 17 12 22 22 17" /><polyline points="2 12 12 17 22 12" /></Icon>;
}
function PowerIcon(p) {
  return <Icon {...p}><path d="M18.36 6.64a9 9 0 1 1-12.73 0" /><line x1="12" y1="2" x2="12" y2="12" /></Icon>;
}
function ChevronRightIcon(p) {
  return <Icon {...p}><polyline points="9 18 15 12 9 6" /></Icon>;
}
function SlidersHorizontalIcon(p) {
  return <Icon {...p}><line x1="21" y1="4" x2="14" y2="4" /><line x1="10" y1="4" x2="3" y2="4" /><line x1="21" y1="12" x2="12" y2="12" /><line x1="8" y1="12" x2="3" y2="12" /><line x1="21" y1="20" x2="16" y2="20" /><line x1="12" y1="20" x2="3" y2="20" /><line x1="14" y1="2" x2="14" y2="6" /><line x1="8" y1="10" x2="8" y2="14" /><line x1="16" y1="18" x2="16" y2="22" /></Icon>;
}
function AlertCircleIcon(p) {
  return <Icon {...p}><circle cx="12" cy="12" r="10" /><line x1="12" y1="8" x2="12" y2="12" /><line x1="12" y1="16" x2="12.01" y2="16" /></Icon>;
}
function Maximize2Icon(p) {
  return <Icon {...p}><polyline points="15 3 21 3 21 9" /><polyline points="9 21 3 21 3 15" /><line x1="21" y1="3" x2="14" y2="10" /><line x1="3" y1="21" x2="10" y2="14" /></Icon>;
}
function CubeIcon(p) {
  return <Icon {...p}><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" /><polyline points="3.27 6.96 12 12.01 20.73 6.96" /><line x1="12" y1="22.08" x2="12" y2="12" /></Icon>;
}
function ZapIcon(p) {
  return <Icon {...p}><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" /></Icon>;
}


// ── Pipeline Node Definitions ─────────────────────────────

const INITIAL_NODES = [
  { id: "vector_memory",    label: "Vector Memory",      icon: DatabaseIcon,      active: true, status: "idle" },
  { id: "semantic_planner", label: "Semantic Planner",    icon: BrainCircuitIcon,  active: true, status: "idle" },
  { id: "motion_engine",    label: "Motion Engine",       icon: ActivityIcon,       active: true, status: "idle" },
  { id: "critic_cascade",   label: "Critic Cascade",      icon: LayersIcon,        active: true, status: "idle" },
  { id: "fastmcp_blender",  label: "FastMCP Blender",     icon: MonitorIcon,       active: true, status: "idle" },
];

const NODE_LOG_MESSAGES = {
  vector_memory:    ["Querying Milvus vector store...", "✓ Retrieved 12 motion embeddings (cosine ≥ 0.87)"],
  semantic_planner: ["Decomposing prompt into motion primitives...", "✓ Planned 4 motion segments via Ollama/LLaMA-3"],
  motion_engine:    ["Running HY-Motion diffusion inference...", "✓ Generated 196-frame motion tensor [1×196×263]"],
  critic_cascade:   ["Evaluating physical plausibility...", "✓ Critic score: 0.94 — PASS (threshold: 0.80)"],
  fastmcp_blender:  ["Connecting to Blender MCP bridge on :9876...", "✓ Exported final_animation.fbx (16.0 MB)"],
};


// ── Camera orbit helpers ──────────────────────────────────

// Convert spherical (theta, phi, radius) + target → camera position
function applyOrbit(camera, o) {
  const sinPhi = Math.sin(o.phi);
  camera.position.set(
    o.target.x + o.radius * sinPhi * Math.sin(o.theta),
    o.target.y + o.radius * Math.cos(o.phi),
    o.target.z + o.radius * sinPhi * Math.cos(o.theta)
  );
  camera.lookAt(o.target);
}

// ── Three.js Viewport Component ───────────────────────────

function ThreeViewport({ isRunning, fbxPath, onFbxStatus }) {
  const mountRef  = useRef(null);
  const sceneRef  = useRef({});
  // Orbit state — lives outside React so the render loop can read it without re-renders
  const orbitRef  = useRef(null);

  // Initialize scene once
  useEffect(() => {
    if (!mountRef.current || typeof THREE === "undefined") return;

    const container = mountRef.current;
    const width  = container.clientWidth;
    const height = container.clientHeight;

    // ── Scene setup ────────────────────────────────────────
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a0a);
    scene.fog = new THREE.FogExp2(0x0a0a0a, 0.018);

    const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 200);

    // Initial orbit state
    const orbit = {
      theta:  Math.PI * 0.3,
      phi:    Math.PI * 0.32,
      radius: 9,
      target: new THREE.Vector3(0, 1.2, 0),
    };
    orbitRef.current = orbit;
    applyOrbit(camera, orbit);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    container.appendChild(renderer.domElement);

    // Grid / lights
    scene.add(new THREE.GridHelper(20, 40, 0x1a1a2e, 0x111122));
    scene.add(new THREE.AmbientLight(0x404060, 0.6));
    const light1 = new THREE.PointLight(0xa3e635, 1.2, 20);
    light1.position.set(3, 5, 3);
    scene.add(light1);
    const light2 = new THREE.PointLight(0x6366f1, 0.8, 20);
    light2.position.set(-3, 3, -3);
    scene.add(light2);

    // Default skeleton
    const skeletonGroup = buildDefaultSkeleton();
    scene.add(skeletonGroup);

    // Particles
    const particleCount = 200;
    const particleGeo   = new THREE.BufferGeometry();
    const pPositions    = new Float32Array(particleCount * 3);
    for (let i = 0; i < particleCount; i++) {
      pPositions[i * 3]     = (Math.random() - 0.5) * 12;
      pPositions[i * 3 + 1] = Math.random() * 6;
      pPositions[i * 3 + 2] = (Math.random() - 0.5) * 12;
    }
    particleGeo.setAttribute("position", new THREE.BufferAttribute(pPositions, 3));
    const particleMat = new THREE.PointsMaterial({ color: 0xa3e635, size: 0.03, transparent: true, opacity: 0.5 });
    const particles   = new THREE.Points(particleGeo, particleMat);
    scene.add(particles);

    const mixer = { current: null };
    const clock = new THREE.Clock();

    sceneRef.current = { scene, camera, renderer, skeletonGroup, particles, particleMat, mixer, fbxModel: null };

    // ── Render loop ────────────────────────────────────────
    let frameId;
    const animate = () => {
      frameId = requestAnimationFrame(animate);
      const delta = clock.getDelta();
      const t     = clock.getElapsedTime();

      if (mixer.current) mixer.current.update(delta);

      // Skeleton idle animation (breathing + arm swing) — no auto-rotation
      if (!sceneRef.current.fbxModel && skeletonGroup.userData.parts) {
        const p = skeletonGroup.userData.parts;
        const breathe = Math.sin(t * 1.5) * 0.02;
        p.head.position.y         = 2.6  + breathe;
        p.torso.position.y        = 1.85 + breathe * 0.5;
        p.leftUpperArm.rotation.z =  0.3 + Math.sin(t * 0.8) * 0.1;
        p.rightUpperArm.rotation.z = -0.3 - Math.sin(t * 0.8) * 0.1;
        p.leftHand.position.y     = 1.05 + Math.sin(t * 0.8) * 0.05;
        p.rightHand.position.y    = 1.05 - Math.sin(t * 0.8) * 0.05;
      }

      if (sceneRef.current.running) {
        const pulse = (Math.sin(t * 4) + 1) * 0.5;
        particleMat.opacity = 0.3 + pulse * 0.4;
      } else {
        particleMat.opacity = 0.5;
      }

      particles.rotation.y = t * 0.05;
      renderer.render(scene, camera);
    };
    animate();

    // ── Camera Controls ────────────────────────────────────
    // Convention (Blender + Maya hybrid):
    //   Orbit  : MMB drag  |  Alt + LMB drag
    //   Pan    : Shift + MMB drag  |  Alt + MMB drag
    //   Zoom   : Scroll wheel  |  Alt + RMB drag
    //   Numpad1: Front view   Numpad3: Right view   Numpad7: Top view
    //   F / Numpad0: Reset / frame

    const drag = { active: false, button: -1, lastX: 0, lastY: 0 };

    // What mode is this drag?
    const dragMode = (button, altKey, shiftKey) => {
      if (button === 1) return shiftKey ? "pan" : "orbit";   // MMB
      if (button === 0 && altKey) return "orbit";             // Alt+LMB
      if (button === 1 && altKey) return "pan";               // Alt+MMB  (already caught above but explicit)
      if (button === 2 && altKey) return "zoom";              // Alt+RMB
      return null;
    };

    const onPointerDown = (e) => {
      const mode = dragMode(e.button, e.altKey, e.shiftKey);
      if (!mode) return;
      e.preventDefault();
      drag.active = true;
      drag.button = e.button;
      drag.altKey  = e.altKey;
      drag.shiftKey = e.shiftKey;
      drag.mode   = mode;
      drag.lastX  = e.clientX;
      drag.lastY  = e.clientY;
      container.setPointerCapture(e.pointerId);
      container.style.cursor = mode === "pan" ? "move" : "grabbing";
    };

    const onPointerMove = (e) => {
      if (!drag.active) return;
      const dx = e.clientX - drag.lastX;
      const dy = e.clientY - drag.lastY;
      drag.lastX = e.clientX;
      drag.lastY = e.clientY;

      const o = orbitRef.current;

      if (drag.mode === "orbit") {
        o.theta -= dx * 0.007;
        o.phi    = Math.max(0.02, Math.min(Math.PI - 0.02, o.phi + dy * 0.007));
      } else if (drag.mode === "pan") {
        // Compute camera right and up vectors from orbit state
        const forward = new THREE.Vector3(
          Math.sin(o.phi) * Math.sin(o.theta),
          Math.cos(o.phi),
          Math.sin(o.phi) * Math.cos(o.theta)
        ).negate();
        const right = new THREE.Vector3().crossVectors(forward, new THREE.Vector3(0, 1, 0)).normalize();
        const up    = new THREE.Vector3().crossVectors(right, forward).normalize();
        const speed = o.radius * 0.0012;
        o.target.addScaledVector(right, -dx * speed);
        o.target.addScaledVector(up,     dy * speed);
      } else if (drag.mode === "zoom") {
        o.radius = Math.max(0.5, o.radius * (1 + dy * 0.01));
      }

      applyOrbit(camera, o);
    };

    const onPointerUp = (e) => {
      if (!drag.active) return;
      drag.active = false;
      container.releasePointerCapture(e.pointerId);
      container.style.cursor = "default";
    };

    const onWheel = (e) => {
      e.preventDefault();
      const o = orbitRef.current;
      const factor = e.deltaY > 0 ? 1.1 : 0.9;
      o.radius = Math.max(0.5, Math.min(80, o.radius * factor));
      applyOrbit(camera, o);
    };

    // Preset views & reset
    const setPreset = (theta, phi, resetTarget) => {
      const o = orbitRef.current;
      if (resetTarget) o.target.set(0, 1.2, 0);
      o.theta = theta;
      o.phi   = phi;
      applyOrbit(camera, o);
    };

    const frameModel = () => {
      const ctx = sceneRef.current;
      const obj = ctx.fbxModel || ctx.skeletonGroup;
      if (!obj) return;
      const box    = new THREE.Box3().setFromObject(obj);
      const center = box.getCenter(new THREE.Vector3());
      const size   = box.getSize(new THREE.Vector3());
      const o      = orbitRef.current;
      o.target.copy(center);
      o.radius = Math.max(size.x, size.y, size.z) * 1.8;
      applyOrbit(camera, o);
    };

    const onKeyDown = (e) => {
      // Only fire when the viewport is hovered (avoids hijacking text inputs)
      if (!container.matches(":hover") && !drag.active) return;
      switch (e.code) {
        case "Numpad1": setPreset(0,           Math.PI / 2, false); break; // Front
        case "Numpad3": setPreset(Math.PI / 2, Math.PI / 2, false); break; // Right
        case "Numpad7": setPreset(0,           0.01,        false); break; // Top
        case "Numpad9": setPreset(Math.PI,     Math.PI / 2, false); break; // Back
        case "KeyF":
        case "Numpad0": frameModel(); break;
        case "Home":    setPreset(Math.PI * 0.3, Math.PI * 0.32, true);  break; // Reset
        default: break;
      }
    };

    // Context menu suppression so Alt+RMB doesn't open it
    const onContextMenu = (e) => e.preventDefault();

    container.addEventListener("pointerdown",  onPointerDown);
    container.addEventListener("pointermove",  onPointerMove);
    container.addEventListener("pointerup",    onPointerUp);
    container.addEventListener("wheel",        onWheel, { passive: false });
    container.addEventListener("contextmenu",  onContextMenu);
    window.addEventListener("keydown",         onKeyDown);

    // ── Resize ─────────────────────────────────────────────
    const onResize = () => {
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener("resize", onResize);

    return () => {
      cancelAnimationFrame(frameId);
      container.removeEventListener("pointerdown",  onPointerDown);
      container.removeEventListener("pointermove",  onPointerMove);
      container.removeEventListener("pointerup",    onPointerUp);
      container.removeEventListener("wheel",        onWheel);
      container.removeEventListener("contextmenu",  onContextMenu);
      window.removeEventListener("keydown",         onKeyDown);
      window.removeEventListener("resize",          onResize);
      if (container.contains(renderer.domElement)) container.removeChild(renderer.domElement);
      renderer.dispose();
    };
  }, []);

  // Track isRunning
  useEffect(() => {
    sceneRef.current.running = isRunning;
  }, [isRunning]);

  // Load FBX when fbxPath changes
  useEffect(() => {
    const ctx = sceneRef.current;
    if (!ctx.scene) return;

    if (ctx.fbxModel) {
      ctx.scene.remove(ctx.fbxModel);
      ctx.fbxModel = null;
      ctx.mixer.current = null;
    }

    if (!fbxPath) {
      ctx.skeletonGroup.visible = true;
      if (onFbxStatus) onFbxStatus(null);
      // Reset orbit to default
      if (orbitRef.current) {
        orbitRef.current.target.set(0, 1.2, 0);
        orbitRef.current.theta  = Math.PI * 0.3;
        orbitRef.current.phi    = Math.PI * 0.32;
        orbitRef.current.radius = 9;
        applyOrbit(ctx.camera, orbitRef.current);
      }
      return;
    }

    ctx.skeletonGroup.visible = false;
    if (onFbxStatus) onFbxStatus("loading");

    const loader = new THREE.FBXLoader();
    loader.load(
      "/assets/" + fbxPath,
      (fbx) => {
        // Auto-scale
        const box    = new THREE.Box3().setFromObject(fbx);
        const size   = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        if (maxDim > 0) fbx.scale.setScalar(3.5 / maxDim);

        // Center on ground
        const scaledBox = new THREE.Box3().setFromObject(fbx);
        const center    = scaledBox.getCenter(new THREE.Vector3());
        fbx.position.x -= center.x;
        fbx.position.z -= center.z;
        fbx.position.y -= scaledBox.min.y;

        fbx.traverse((child) => {
          if (child.isMesh) {
            child.castShadow = child.receiveShadow = true;
            if (child.material) {
              const mats = Array.isArray(child.material) ? child.material : [child.material];
              mats.forEach(m => {
                if (m.emissive) m.emissive.set(0x1a1a2e);
                m.roughness = Math.min(m.roughness || 0.5, 0.7);
              });
            }
          }
        });

        ctx.scene.add(fbx);
        ctx.fbxModel = fbx;

        if (fbx.animations && fbx.animations.length > 0) {
          const mixer = new THREE.AnimationMixer(fbx);
          fbx.animations.forEach(clip => {
            const action = mixer.clipAction(clip);
            action.setLoop(THREE.LoopRepeat);
            action.play();
          });
          ctx.mixer.current = mixer;
          if (onFbxStatus) onFbxStatus("playing");
        } else {
          if (onFbxStatus) onFbxStatus("static");
        }

        // Frame the loaded model by updating orbit state (not camera directly)
        const finalBox    = new THREE.Box3().setFromObject(fbx);
        const finalCenter = finalBox.getCenter(new THREE.Vector3());
        const finalSize   = finalBox.getSize(new THREE.Vector3());
        const dist        = Math.max(finalSize.x, finalSize.y, finalSize.z) * 1.8;

        if (orbitRef.current) {
          const o = orbitRef.current;
          o.target.copy(finalCenter);
          o.radius = dist;
          // Keep current theta/phi so the angle feels natural
          applyOrbit(ctx.camera, o);
        }
      },
      undefined,
      (err) => {
        console.error("FBX load error:", err);
        ctx.skeletonGroup.visible = true;
        if (onFbxStatus) onFbxStatus("error");
      }
    );
  }, [fbxPath, onFbxStatus]);

  return <div ref={mountRef} className="w-full h-full viewport-canvas" />;
}

// Builds the default geometric skeleton shown when no FBX is loaded
function buildDefaultSkeleton() {
  const group = new THREE.Group();
  const matLime = new THREE.MeshStandardMaterial({ color: 0xa3e635, emissive: 0x4d7c0f, roughness: 0.3, metalness: 0.7 });
  const matDark = new THREE.MeshStandardMaterial({ color: 0x404040, emissive: 0x1a1a2e, roughness: 0.5, metalness: 0.6 });
  const matAccent = new THREE.MeshStandardMaterial({ color: 0x6366f1, emissive: 0x3730a3, roughness: 0.3, metalness: 0.7 });

  const head = new THREE.Mesh(new THREE.SphereGeometry(0.25, 16, 16), matLime);
  head.position.y = 2.6;
  group.add(head);

  const torso = new THREE.Mesh(new THREE.BoxGeometry(0.6, 0.9, 0.3), matDark);
  torso.position.y = 1.85;
  group.add(torso);

  const pelvis = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.25, 0.25), matAccent);
  pelvis.position.y = 1.3;
  group.add(pelvis);

  const armGeo = new THREE.CylinderGeometry(0.06, 0.05, 0.7, 8);
  const leftUpperArm = new THREE.Mesh(armGeo, matLime);
  leftUpperArm.position.set(-0.5, 2.0, 0); leftUpperArm.rotation.z = 0.3;
  group.add(leftUpperArm);
  const rightUpperArm = new THREE.Mesh(armGeo, matLime);
  rightUpperArm.position.set(0.5, 2.0, 0); rightUpperArm.rotation.z = -0.3;
  group.add(rightUpperArm);
  const leftLowerArm = new THREE.Mesh(armGeo, matAccent);
  leftLowerArm.position.set(-0.7, 1.4, 0); leftLowerArm.rotation.z = 0.15;
  group.add(leftLowerArm);
  const rightLowerArm = new THREE.Mesh(armGeo, matAccent);
  rightLowerArm.position.set(0.7, 1.4, 0); rightLowerArm.rotation.z = -0.15;
  group.add(rightLowerArm);

  const handGeo = new THREE.SphereGeometry(0.08, 8, 8);
  const leftHand = new THREE.Mesh(handGeo, matLime);
  leftHand.position.set(-0.78, 1.05, 0);
  group.add(leftHand);
  const rightHand = new THREE.Mesh(handGeo, matLime);
  rightHand.position.set(0.78, 1.05, 0);
  group.add(rightHand);

  const legGeo = new THREE.CylinderGeometry(0.07, 0.06, 0.75, 8);
  [[matDark, -0.18, 0.85], [matDark, 0.18, 0.85], [matAccent, -0.18, 0.2], [matAccent, 0.18, 0.2]].forEach(([m, x, y]) => {
    const leg = new THREE.Mesh(legGeo, m);
    leg.position.set(x, y, 0);
    group.add(leg);
  });

  const footGeo = new THREE.BoxGeometry(0.12, 0.06, 0.22);
  [[-0.18, 0.03, 0.05], [0.18, 0.03, 0.05]].forEach(([x, y, z]) => {
    const foot = new THREE.Mesh(footGeo, matLime);
    foot.position.set(x, y, z);
    group.add(foot);
  });

  const jointGeo = new THREE.SphereGeometry(0.07, 8, 8);
  [[0,2.3,0],[-0.3,2.3,0],[0.3,2.3,0],[-0.18,1.2,0],[0.18,1.2,0],[-0.18,0.5,0],[0.18,0.5,0],[-0.5,1.65,0],[0.5,1.65,0]].forEach(([x,y,z]) => {
    const j = new THREE.Mesh(jointGeo, matLime);
    j.position.set(x, y, z);
    group.add(j);
  });

  // Store named refs for idle animation
  group.userData.parts = { head, torso, leftUpperArm, rightUpperArm, leftHand, rightHand };
  return group;
}


// ── Pipeline Node Component ───────────────────────────────

function PipelineNode({ node, index, total, onToggle }) {
  const NodeIcon = node.icon;
  const statusColor =
    node.status === "running" ? "text-amber-400" :
    node.status === "success" ? "text-lime-400" :
    "text-neutral-600";

  const StatusIcon =
    node.status === "running" ? ActivityIcon :
    node.status === "success" ? CheckCircle2Icon :
    ChevronRightIcon;

  return (
    <div className="relative flex items-center gap-3 group">
      {/* Vertical line */}
      {index < total - 1 && (
        <div className="absolute left-[17px] top-[36px] w-[2px] h-[calc(100%+4px)] bg-neutral-800 z-0" />
      )}

      {/* Node dot */}
      <div className={`relative z-10 w-[36px] h-[36px] rounded-lg flex items-center justify-center border transition-all duration-300 ${
        node.status === "running"
          ? "bg-amber-400/10 border-amber-400/40 shadow-[0_0_12px_rgba(251,191,36,0.15)]"
          : node.status === "success"
          ? "bg-lime-400/10 border-lime-400/40 shadow-[0_0_12px_rgba(163,230,53,0.15)]"
          : node.active
          ? "bg-neutral-800/80 border-neutral-700"
          : "bg-neutral-900/50 border-neutral-800"
      }`}>
        <NodeIcon size={16} className={
          node.status === "running" ? "text-amber-400" :
          node.status === "success" ? "text-lime-400" :
          node.active ? "text-neutral-400" : "text-neutral-700"
        } />
      </div>

      {/* Label + status */}
      <div className="flex-1 min-w-0">
        <div className={`text-xs font-medium truncate transition-colors ${
          node.active ? "text-neutral-300" : "text-neutral-600 line-through"
        }`}>
          {node.label}
        </div>
        <div className={`flex items-center gap-1 mt-0.5 ${statusColor}`}>
          <StatusIcon size={10} />
          <span className="text-[10px] uppercase tracking-wider font-mono">
            {node.status === "running" ? "processing" : node.status === "success" ? "complete" : node.active ? "ready" : "bypassed"}
          </span>
        </div>
      </div>

      {/* Power toggle */}
      <button
        onClick={() => onToggle(node.id)}
        className={`p-1.5 rounded-md transition-all opacity-0 group-hover:opacity-100 focus:opacity-100 ${
          node.active
            ? "text-lime-400 hover:bg-lime-400/10"
            : "text-neutral-600 hover:bg-neutral-800"
        }`}
        title={node.active ? "Bypass node" : "Enable node"}
      >
        <PowerIcon size={13} />
      </button>
    </div>
  );
}


// ── Terminal Console Component ────────────────────────────

function TerminalConsole({ logs }) {
  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <div className="h-full flex flex-col bg-neutral-950 border-t border-neutral-800">
      {/* Terminal header */}
      <div className="flex items-center gap-2 px-3 py-1.5 bg-neutral-900/60 border-b border-neutral-800/50">
        <TerminalIcon size={12} className="text-lime-400" />
        <span className="text-[11px] font-mono text-neutral-500 tracking-wider uppercase">Pipeline Console</span>
        <div className="flex-1" />
        <span className="text-[10px] font-mono text-neutral-600">{logs.length} entries</span>
      </div>

      {/* Log output */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-3 py-2 font-mono text-[11px] leading-[1.7]">
        {logs.length === 0 ? (
          <div className="text-neutral-700 italic">Awaiting pipeline execution...</div>
        ) : (
          logs.map((log, i) => (
            <div key={i} className={`whitespace-pre-wrap ${
              log.includes("✓") || log.includes("SUCCESS") || log.includes("COMPLETE")
                ? "text-lime-400"
                : log.includes("⏩") || log.includes("BYPASS")
                ? "text-amber-400"
                : log.includes("ERROR") || log.includes("✗")
                ? "text-red-400"
                : log.includes("──") || log.includes("═")
                ? "text-neutral-600"
                : "text-neutral-400"
            }`}>
              {log}
            </div>
          ))
        )}
      </div>
    </div>
  );
}


// ── Main App Component ────────────────────────────────────

function App() {
  const [prompt, setPrompt] = useState("A person walks forward, waves hello, then sits down on a chair");
  const [isRunning, setIsRunning] = useState(false);
  const [logs, setLogs] = useState([]);
  const [nodes, setNodes] = useState(INITIAL_NODES);
  const [settings, setSettings] = useState({
    duration: 4.0,
    engine: "hy-motion-v1",
    project: "default",
    criticThreshold: 0.80,
    kinematicChecks: true,
  });
  const [mcpConnected, setMcpConnected] = useState(true);
  const [fbxFiles, setFbxFiles] = useState([]);
  const [fbxFilesLoading, setFbxFilesLoading] = useState(true);
  const [selectedFbx, setSelectedFbx] = useState("");
  const [fbxStatus, setFbxStatus] = useState(null); // null | "loading" | "playing" | "static" | "error"
  const abortRef = useRef(false);

  // Fetch available FBX files from the server API
  useEffect(() => {
    setFbxFilesLoading(true);
    fetch("/api/fbx-files")
      .then(r => r.json())
      .then(files => { setFbxFiles(files); setFbxFilesLoading(false); })
      .catch(() => { setFbxFiles([]); setFbxFilesLoading(false); });
  }, []);

  const addLog = useCallback((msg) => {
    const ts = new Date().toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });
    setLogs(prev => [...prev, `[${ts}]  ${msg}`]);
  }, []);

  const toggleNode = useCallback((id) => {
    if (isRunning) return;
    setNodes(prev => prev.map(n => n.id === id ? { ...n, active: !n.active } : n));
  }, [isRunning]);

  const updateSetting = useCallback((key, value) => {
    setSettings(prev => ({ ...prev, [key]: value }));
  }, []);

  const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

  const runPipeline = useCallback(async () => {
    if (isRunning) {
      // Abort
      abortRef.current = true;
      setIsRunning(false);
      addLog("✗ Pipeline ABORTED by user");
      addLog("─────────────────────────────────────────");
      setNodes(prev => prev.map(n => ({ ...n, status: "idle" })));
      return;
    }

    abortRef.current = false;
    setIsRunning(true);
    setLogs([]);

    addLog("═══════════════════════════════════════════");
    addLog("  GenAnim Pipeline — Execution Started");
    addLog("═══════════════════════════════════════════");
    addLog(`  Engine:   ${settings.engine}`);
    addLog(`  Duration: ${settings.duration}s @ 30fps (${Math.round(settings.duration * 30)} frames)`);
    addLog(`  Project:  ${settings.project}`);
    addLog(`  Critic:   threshold=${settings.criticThreshold.toFixed(2)}`);
    addLog(`  Kinem.:   ${settings.kinematicChecks ? "enabled" : "disabled"}`);
    addLog(`  Prompt:   "${prompt}"`);
    addLog("─────────────────────────────────────────");

    // Reset all nodes
    setNodes(prev => prev.map(n => ({ ...n, status: "idle" })));

    for (let i = 0; i < INITIAL_NODES.length; i++) {
      if (abortRef.current) break;

      const nodeId = INITIAL_NODES[i].id;
      const nodeLabel = INITIAL_NODES[i].label;

      // Check current active state
      const currentNodes = await new Promise(resolve => {
        setNodes(prev => { resolve(prev); return prev; });
      });
      const currentNode = currentNodes.find(n => n.id === nodeId);

      if (!currentNode.active) {
        addLog(`⏩ BYPASSED: ${nodeLabel}`);
        setNodes(prev => prev.map(n => n.id === nodeId ? { ...n, status: "idle" } : n));
        await sleep(300);
        continue;
      }

      // Set running
      addLog(`▶ Starting: ${nodeLabel}...`);
      setNodes(prev => prev.map(n => n.id === nodeId ? { ...n, status: "running" } : n));

      const messages = NODE_LOG_MESSAGES[nodeId];
      await sleep(600);
      if (abortRef.current) break;
      addLog(`  ${messages[0]}`);

      await sleep(900 + Math.random() * 600);
      if (abortRef.current) break;

      addLog(`  ${messages[1]}`);
      setNodes(prev => prev.map(n => n.id === nodeId ? { ...n, status: "success" } : n));
      await sleep(200);
    }

    if (!abortRef.current) {
      addLog("─────────────────────────────────────────");
      addLog("✓ PIPELINE COMPLETE — All stages finished successfully");
      addLog(`✓ Output: final_animation.fbx`);
      addLog("═══════════════════════════════════════════");
    }

    setIsRunning(false);

    // Reset nodes after a delay
    setTimeout(() => {
      setNodes(prev => prev.map(n => ({ ...n, status: "idle" })));
    }, 3000);
  }, [isRunning, settings, prompt, addLog]);

  return (
    <div className="h-screen w-screen flex flex-col bg-neutral-950 text-neutral-300 overflow-hidden">

      {/* ── Top Navigation Bar ─────────────────────── */}
      <header className="flex items-center gap-3 px-4 py-2.5 bg-neutral-900/70 border-b border-neutral-800 shrink-0">
        {/* Logo / Title */}
        <div className="flex items-center gap-2 shrink-0">
          <CubeIcon size={20} className="text-lime-400" />
          <h1 className="text-sm font-bold tracking-tight text-neutral-100">
            GenAnim<span className="text-lime-400">Dashboard</span>
          </h1>
        </div>

        <div className="w-px h-6 bg-neutral-800 mx-1" />

        {/* Prompt input */}
        <div className="flex-1 relative">
          <input
            type="text"
            value={prompt}
            onChange={e => setPrompt(e.target.value)}
            placeholder="Describe the motion you want to generate..."
            className="w-full bg-neutral-800/60 border border-neutral-700/50 rounded-lg px-3 py-1.5 text-sm text-neutral-200 placeholder-neutral-600 focus:outline-none focus:border-lime-400/40 focus:ring-1 focus:ring-lime-400/20 transition-all"
            disabled={isRunning}
          />
        </div>

        <div className="w-px h-6 bg-neutral-800 mx-1" />

        {/* MCP Status */}
        <div className="flex items-center gap-1.5 shrink-0 px-2 py-1 rounded-md bg-neutral-800/40 border border-neutral-800">
          <div className={`w-2 h-2 rounded-full ${mcpConnected ? "bg-lime-400 shadow-[0_0_6px_rgba(163,230,53,0.4)]" : "bg-red-400 shadow-[0_0_6px_rgba(248,113,113,0.4)]"}`} />
          <span className="text-[10px] font-mono text-neutral-500 uppercase tracking-wider">MCP</span>
          <span className={`text-[10px] font-mono ${mcpConnected ? "text-lime-400" : "text-red-400"}`}>
            {mcpConnected ? "LIVE" : "DOWN"}
          </span>
        </div>

        {/* Generate / Abort Button */}
        <button
          onClick={runPipeline}
          className={`flex items-center gap-1.5 px-4 py-1.5 rounded-lg text-xs font-semibold transition-all shrink-0 ${
            isRunning
              ? "bg-red-500/20 text-red-400 border border-red-500/30 hover:bg-red-500/30"
              : "bg-lime-400/15 text-lime-400 border border-lime-400/30 hover:bg-lime-400/25 hover:shadow-[0_0_20px_rgba(163,230,53,0.1)]"
          }`}
        >
          {isRunning ? <SquareIcon size={13} /> : <PlayIcon size={13} />}
          {isRunning ? "Abort" : "Generate"}
        </button>
      </header>

      {/* ── Main Workspace ─────────────────────────── */}
      <div className="flex-1 flex overflow-hidden min-h-0">

        {/* Left Panel — Graph Topology */}
        <aside className="w-[220px] shrink-0 bg-neutral-900/40 border-r border-neutral-800/50 flex flex-col">
          <div className="flex items-center gap-2 px-3 py-2 border-b border-neutral-800/50">
            <ZapIcon size={12} className="text-lime-400" />
            <span className="text-[11px] font-semibold text-neutral-400 uppercase tracking-wider">LangGraph Nodes</span>
          </div>
          <div className="flex-1 overflow-y-auto p-3 space-y-3">
            {nodes.map((node, i) => (
              <PipelineNode
                key={node.id}
                node={node}
                index={i}
                total={nodes.length}
                onToggle={toggleNode}
              />
            ))}
          </div>
          <div className="px-3 py-2 border-t border-neutral-800/50">
            <div className="text-[10px] font-mono text-neutral-600">
              {nodes.filter(n => n.active).length}/{nodes.length} nodes active
            </div>
          </div>
        </aside>

        {/* Center Panel — 3D Viewport */}
        <main className="flex-1 relative min-w-0 bg-neutral-950">
          {/* Top bar: FBX selector + status */}
          <div className="absolute top-2 left-3 right-3 z-10 flex items-center gap-2">
            <span className="text-[10px] font-mono text-neutral-600 bg-neutral-950/80 px-1.5 py-0.5 rounded shrink-0">
              VIEWPORT
            </span>
            <select
              value={selectedFbx}
              onChange={e => setSelectedFbx(e.target.value)}
              aria-label="Select FBX file"
              disabled={fbxFilesLoading}
              className={`border rounded-md px-2 py-1 text-[11px] font-mono focus:outline-none min-w-0 max-w-[280px] truncate transition-colors ${
                fbxFilesLoading
                  ? "bg-neutral-900/50 border-neutral-800 text-neutral-600 cursor-not-allowed"
                  : "bg-neutral-900/90 border-neutral-700/50 text-neutral-300 focus:border-lime-400/40"
              }`}
            >
              <option value="">{fbxFilesLoading ? "⟳ Scanning for FBX files…" : "— no FBX loaded —"}</option>
              {fbxFiles.map(f => (
                <option key={f} value={f}>{f.split("/").pop()} — {f}</option>
              ))}
            </select>
            {fbxFilesLoading && (
              <span className="text-[10px] font-mono text-neutral-500 bg-neutral-800/60 px-1.5 py-0.5 rounded animate-pulse shrink-0">
                scanning…
              </span>
            )}
            {fbxStatus === "loading" && (
              <span className="text-[10px] font-mono text-amber-400 bg-amber-400/10 px-1.5 py-0.5 rounded animate-pulse shrink-0">
                ● LOADING
              </span>
            )}
            {fbxStatus === "playing" && (
              <span className="text-[10px] font-mono text-lime-400 bg-lime-400/10 px-1.5 py-0.5 rounded shrink-0">
                ▶ PLAYING
              </span>
            )}
            {fbxStatus === "static" && (
              <span className="text-[10px] font-mono text-neutral-400 bg-neutral-800/80 px-1.5 py-0.5 rounded shrink-0">
                ■ STATIC
              </span>
            )}
            {fbxStatus === "error" && (
              <span className="text-[10px] font-mono text-red-400 bg-red-400/10 px-1.5 py-0.5 rounded shrink-0">
                ✗ ERROR
              </span>
            )}
            {isRunning && (
              <span className="text-[10px] font-mono text-amber-400 bg-amber-400/10 px-1.5 py-0.5 rounded animate-pulse shrink-0">
                ● RUNNING
              </span>
            )}
            <div className="flex-1" />
            <button type="button" aria-label="Maximize viewport" title="Maximize viewport" className="p-1 text-neutral-600 hover:text-neutral-400 transition-colors shrink-0">
              <Maximize2Icon size={13} />
            </button>
          </div>
          <ThreeViewport isRunning={isRunning} fbxPath={selectedFbx} onFbxStatus={setFbxStatus} />
          {/* Camera controls hint — bottom-left corner */}
          <div className="absolute bottom-2 left-3 z-10 flex flex-col gap-0.5 pointer-events-none">
            {[
              ["MMB drag / Alt+LMB", "Orbit"],
              ["Shift+MMB / Alt+MMB", "Pan"],
              ["Scroll / Alt+RMB",    "Zoom"],
              ["Num 1/3/7  F  Home",  "Views / Frame / Reset"],
            ].map(([keys, label]) => (
              <div key={keys} className="flex items-center gap-1.5">
                <span className="text-[9px] font-mono text-neutral-700 bg-neutral-900/60 px-1 py-px rounded">{keys}</span>
                <span className="text-[9px] text-neutral-700">{label}</span>
              </div>
            ))}
          </div>
        </main>

        {/* Right Panel — Module Parameters */}
        <aside className="w-[240px] shrink-0 bg-neutral-900/40 border-l border-neutral-800/50 flex flex-col">
          <div className="flex items-center gap-2 px-3 py-2 border-b border-neutral-800/50">
            <SlidersHorizontalIcon size={12} className="text-lime-400" />
            <span className="text-[11px] font-semibold text-neutral-400 uppercase tracking-wider">Parameters</span>
          </div>
          <div className="flex-1 overflow-y-auto p-3 space-y-4">

            {/* Duration slider */}
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label className="text-[11px] text-neutral-400 font-medium">Duration</label>
                <span className="text-[11px] font-mono text-lime-400">{settings.duration.toFixed(1)}s</span>
              </div>
              <input
                type="range"
                min="0.5"
                max="15"
                step="0.5"
                value={settings.duration}
                onChange={e => updateSetting("duration", parseFloat(e.target.value))}
                aria-label="Duration"
                className="w-full"
                disabled={isRunning}
              />
              <div className="flex justify-between mt-0.5">
                <span className="text-[9px] text-neutral-700 font-mono">0.5s</span>
                <span className="text-[9px] text-neutral-700 font-mono">{Math.round(settings.duration * 30)}f</span>
                <span className="text-[9px] text-neutral-700 font-mono">15s</span>
              </div>
            </div>

            {/* Critic Threshold slider */}
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label className="text-[11px] text-neutral-400 font-medium">Vision AI Threshold</label>
                <span className="text-[11px] font-mono text-lime-400">{settings.criticThreshold.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={settings.criticThreshold}
                onChange={e => updateSetting("criticThreshold", parseFloat(e.target.value))}
                aria-label="Vision AI Threshold"
                className="w-full"
                disabled={isRunning}
              />
              <div className="flex justify-between mt-0.5">
                <span className="text-[9px] text-neutral-700 font-mono">0.00</span>
                <span className="text-[9px] text-neutral-700 font-mono">1.00</span>
              </div>
            </div>

            <div className="w-full h-px bg-neutral-800/50" />

            {/* Project Partition dropdown */}
            <div>
              <label className="text-[11px] text-neutral-400 font-medium block mb-1.5">Project Partition</label>
              <select
                value={settings.project}
                onChange={e => updateSetting("project", e.target.value)}
                aria-label="Project Partition"
                className="w-full bg-neutral-800/60 border border-neutral-700/50 rounded-md px-2.5 py-1.5 text-xs text-neutral-200 focus:outline-none focus:border-lime-400/40"
                disabled={isRunning}
              >
                <option value="default">default</option>
                <option value="cinematic">cinematic</option>
                <option value="game-anim">game-anim</option>
                <option value="mocap-retarget">mocap-retarget</option>
              </select>
            </div>

            {/* Diffusion Model */}
            <div>
              <label className="text-[11px] text-neutral-400 font-medium block mb-1.5">Diffusion Model</label>
              <div className="grid grid-cols-1 gap-1">
                {["hy-motion-v1", "mdm-base", "mld-v2"].map(engine => (
                  <button
                    key={engine}
                    onClick={() => !isRunning && updateSetting("engine", engine)}
                    className={`text-[11px] font-mono px-2.5 py-1.5 rounded-md border transition-all text-left ${
                      settings.engine === engine
                        ? "bg-lime-400/10 border-lime-400/30 text-lime-400"
                        : "bg-neutral-800/30 border-neutral-800 text-neutral-500 hover:border-neutral-700 hover:text-neutral-400"
                    }`}
                    disabled={isRunning}
                  >
                    {engine}
                  </button>
                ))}
              </div>
            </div>

            <div className="w-full h-px bg-neutral-800/50" />

            {/* Kinematic Math checkbox */}
            <div>
              <label className="flex items-center gap-2.5 cursor-pointer group">
                <div className={`w-4 h-4 rounded border flex items-center justify-center transition-all ${
                  settings.kinematicChecks
                    ? "bg-lime-400/20 border-lime-400/50"
                    : "bg-neutral-800/60 border-neutral-700"
                }`}>
                  {settings.kinematicChecks && <CheckCircle2Icon size={10} className="text-lime-400" />}
                </div>
                <span className="text-[11px] text-neutral-400 font-medium group-hover:text-neutral-300 transition-colors">
                  Kinematic Math
                </span>
              </label>
              <p className="text-[9px] text-neutral-600 mt-1 ml-[26px]">
                Physics-aware joint constraint validation
              </p>
            </div>

          </div>
        </aside>
      </div>

      {/* ── Bottom Terminal Console ─────────────────── */}
      <div className="h-[180px] shrink-0">
        <TerminalConsole logs={logs} />
      </div>
    </div>
  );
}

// ── Mount ─────────────────────────────────────────────────

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(React.createElement(App));
