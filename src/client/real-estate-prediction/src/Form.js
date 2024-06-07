import React from 'react';
import { useForm } from 'react-hook-form';

export default function Form({ onUpdate, hasLotArea }) {
  const { register, handleSubmit } = useForm();

  const onSubmit = data => {
    onUpdate({
      bedrooms: parseInt(data.bedrooms),
      bathrooms: parseInt(data.bathrooms),
      livingArea: parseInt(data.livingArea),
      lotArea: hasLotArea ? parseInt(data.lotArea) : null
    });
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <div>
        <label>Bedrooms:</label>
        <input type="number" {...register('bedrooms')} />
      </div>
      <div>
        <label>Bathrooms:</label>
        <input type="number" {...register('bathrooms')} />
      </div>
      <div>
        <label>Living Area (sq ft):</label>
        <input type="number" {...register('livingArea')} />
      </div>
      {hasLotArea && (
        <div>
          <label>Lot Area (sq ft):</label>
          <input type="number" {...register('lotArea')} />
        </div>
      )}
      <button type="submit">Update</button>
    </form>
  );
}
