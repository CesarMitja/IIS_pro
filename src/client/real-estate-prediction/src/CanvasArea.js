import React, { useRef, useEffect } from 'react';

function CanvasArea({ type, setFormData }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    let isDragging = false;
    let rect = {
      x: 50,
      y: 50,
      width: 100,
      height: 100
    };

    const drawRect = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = 'lightblue';
      ctx.fillRect(rect.x, rect.y, rect.width, rect.height);
    };

    const getMousePos = (canvas, evt) => {
      const rect = canvas.getBoundingClientRect();
      return {
        x: evt.clientX - rect.left,
        y: evt.clientY - rect.top
      };
    };

    const isInsideRect = (pos, rect) => {
      return pos.x > rect.x && pos.x < rect.x + rect.width && pos.y > rect.y && pos.y < rect.y + rect.height;
    };

    const onMouseDown = (e) => {
      const mousePos = getMousePos(canvas, e);
      if (isInsideRect(mousePos, rect)) {
        isDragging = true;
      }
    };

    const onMouseMove = (e) => {
      if (isDragging) {
        const mousePos = getMousePos(canvas, e);
        rect.width = Math.max(mousePos.x - rect.x, 10);
        rect.height = Math.max(mousePos.y - rect.y, 10);
        drawRect();
        updateFormData(rect.width, rect.height);
      }
    };

    const onMouseUp = () => {
      isDragging = false;
    };

    const updateFormData = (width, height) => {
      if (type === 'rent') {
        setFormData((prev) => ({
          ...prev,
          LivingArea: Math.round(width * 10) // Multiply to simulate actual area
        }));
      } else if (type === 'price') {
        setFormData((prev) => ({
          ...prev,
          LivingArea: Math.round(width * 10),
          LotArea: Math.round(height * 10)
        }));
      }
    };

    canvas.addEventListener('mousedown', onMouseDown);
    canvas.addEventListener('mousemove', onMouseMove);
    canvas.addEventListener('mouseup', onMouseUp);

    drawRect();

    return () => {
      canvas.removeEventListener('mousedown', onMouseDown);
      canvas.removeEventListener('mousemove', onMouseMove);
      canvas.removeEventListener('mouseup', onMouseUp);
    };
  }, [setFormData, type]);

  return <canvas ref={canvasRef} width={400} height={400} style={{ border: '1px solid black' }} />;
}

export default CanvasArea;
