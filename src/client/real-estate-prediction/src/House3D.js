import React, { useRef } from 'react';
import { Canvas, extend, useFrame } from '@react-three/fiber';
import { OrbitControls, useGLTF } from '@react-three/drei';
import * as THREE from 'three';

// Extend Three.js objects for use with react-three-fiber
extend({ PlaneGeometry: THREE.PlaneGeometry });

function HouseModel({ scale }) {
  const group = useRef();
  const { scene } = useGLTF('/models/house-model.glb');

  // Set material to light blue and enable shadows
  scene.traverse((child) => {
    if (child.isMesh) {
      child.material = new THREE.MeshStandardMaterial({ color: 'white' });
      child.castShadow = true;
      child.receiveShadow = true;
    }
  });

  return (
    <group ref={group} scale={scale}>
      <primitive object={scene} />
    </group>
  );
}

export default function House3D() {
  return (
    <Canvas shadows camera={{ position: [5, 2, 5], fov: 45 }}>
      <ambientLight intensity={0.2} />
      <pointLight position={[10, 10, 10]} intensity={1} castShadow />
      <directionalLight
        position={[5, 5, 5]}
        intensity={2}
        castShadow
        shadow-mapSize-width={1024}
        shadow-mapSize-height={1024}
        shadow-camera-near={0.1}
        shadow-camera-far={50}
      />
      <HouseModel scale={[0.1, 0.1, 0.1]} />
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -1, 0]}>
        <planeGeometry args={[50, 50]} />
        <shadowMaterial opacity={0.3} />
      </mesh>
      <OrbitControls />
    </Canvas>
  );
}
