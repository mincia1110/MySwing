/**
 * Reusable form field component.
 *
 * Renders a label + input/select with optional validation error display.
 * Supports both text/number inputs and select dropdowns.
 *
 * Used by UserProfileForm to render all fields uniformly with
 * accessible label/error association.
 */
import type { ReactNode } from "react";
import "./FormField.css";

export interface FormFieldOption {
  value: string;
  label: string;
}

interface BaseProps {
  /** Field id; used for `htmlFor`/`id` association and test selectors. */
  id: string;
  /** Visible label rendered above the field. */
  label: string;
  /** Helpful hint text rendered below the field (range info, units, etc.). */
  hint?: ReactNode;
  /** Validation error message; when present, renders error styling + alert. */
  error?: string | null;
  /** Whether the field is required (renders the * indicator). */
  required?: boolean;
  /** Disable the input. */
  disabled?: boolean;
}

export interface TextFormFieldProps extends BaseProps {
  type: "text" | "number";
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  /** HTML min/max for number inputs (also enforced via hint/validation). */
  min?: number;
  max?: number;
  step?: number;
  /** Optional unit shown next to the input (e.g., "cm", "oz"). */
  unit?: string;
}

export interface SelectFormFieldProps extends BaseProps {
  type: "select";
  value: string;
  onChange: (value: string) => void;
  options: FormFieldOption[];
  /** Optional placeholder option label (rendered with empty value). */
  placeholder?: string;
}

export type FormFieldProps = TextFormFieldProps | SelectFormFieldProps;

export function FormField(props: FormFieldProps) {
  const errorId = `${props.id}-error`;
  const hintId = `${props.id}-hint`;
  const describedBy = [
    props.hint ? hintId : null,
    props.error ? errorId : null,
  ]
    .filter(Boolean)
    .join(" ") || undefined;

  const wrapperClass = [
    "form-field",
    props.error ? "form-field--error" : "",
    props.disabled ? "form-field--disabled" : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div className={wrapperClass} data-testid={`form-field-${props.id}`}>
      <label htmlFor={props.id} className="form-field__label">
        {props.label}
        {props.required ? (
          <span className="form-field__required" aria-label="required">
            {" *"}
          </span>
        ) : null}
      </label>

      {props.type === "select" ? (
        <select
          id={props.id}
          name={props.id}
          value={props.value}
          onChange={(event) => props.onChange(event.target.value)}
          disabled={props.disabled}
          required={props.required}
          aria-invalid={Boolean(props.error)}
          aria-describedby={describedBy}
          className="form-field__input form-field__select"
          data-testid={`form-field-input-${props.id}`}
        >
          {props.placeholder !== undefined ? (
            <option value="">{props.placeholder}</option>
          ) : null}
          {props.options.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      ) : (
        <div className="form-field__input-row">
          <input
            id={props.id}
            name={props.id}
            type={props.type}
            value={props.value}
            onChange={(event) => props.onChange(event.target.value)}
            placeholder={props.placeholder}
            min={props.min}
            max={props.max}
            step={props.step}
            disabled={props.disabled}
            required={props.required}
            aria-invalid={Boolean(props.error)}
            aria-describedby={describedBy}
            className="form-field__input"
            data-testid={`form-field-input-${props.id}`}
          />
          {props.unit ? (
            <span className="form-field__unit">{props.unit}</span>
          ) : null}
        </div>
      )}

      {props.hint ? (
        <p
          id={hintId}
          className="form-field__hint"
          data-testid={`form-field-hint-${props.id}`}
        >
          {props.hint}
        </p>
      ) : null}

      {props.error ? (
        <p
          id={errorId}
          className="form-field__error"
          role="alert"
          data-testid={`form-field-error-${props.id}`}
        >
          {props.error}
        </p>
      ) : null}
    </div>
  );
}

export default FormField;
